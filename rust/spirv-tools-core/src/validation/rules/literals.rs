//! Literal number validation rules.
//!
//! This module validates that literal numbers in SPIR-V instructions are
//! correctly encoded according to the specification:
//!
//! - The high-order bits of a literal number must be:
//!   - 0 for a floating-point type
//!   - 0 for an integer type with Signedness of 0 (unsigned)
//!   - Sign-extended when Signedness is 1 (signed)
//!
//! This primarily affects integer types with bit widths less than 32 bits,
//! where the upper bits of the 32-bit word must be properly extended.

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};
use crate::validation::ValidationResult;

// ============================================================================
// Literal Encoding Helpers
// ============================================================================

/// Number encoding kind for a literal value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberKind {
    /// Signed integer (upper bits should be sign-extended)
    SignedInt,
    /// Unsigned integer (upper bits should be zero)
    UnsignedInt,
    /// Floating point (upper bits should be zero)
    Float,
}

/// Verifies that the upper bits of a value are correctly extended.
///
/// For values with bit widths less than 32, this checks that:
/// - For unsigned integers and floats: upper bits are zero
/// - For signed integers: upper bits are sign-extended from the value
///
/// Returns `true` if the encoding is valid, `false` otherwise.
fn verify_upper_bits(value: u32, bit_width: u32, kind: NumberKind) -> bool {
    // Bit widths that are a multiple of 32 have no upper bits to check
    if bit_width == 0 || bit_width >= 32 || bit_width.is_multiple_of(32) {
        return true;
    }

    let upper_mask = 0xFFFF_FFFFu32 << bit_width;
    let upper_bits = value & upper_mask;

    match kind {
        NumberKind::SignedInt => {
            // Check sign bit
            let sign_bit = value & (1u32 << (bit_width - 1));
            if sign_bit != 0 {
                // Negative: upper bits should all be 1
                upper_bits == upper_mask
            } else {
                // Positive: upper bits should all be 0
                upper_bits == 0
            }
        }
        NumberKind::UnsignedInt | NumberKind::Float => {
            // Upper bits should be zero
            upper_bits == 0
        }
    }
}

/// Extract the signedness from an OpTypeInt instruction.
///
/// Returns `true` if the integer type is signed, `false` if unsigned.
fn get_int_signedness(inst: &rspirv::dr::Instruction) -> bool {
    // OpTypeInt has: result_type=None, result_id, width (operand 0), signedness (operand 1)
    if inst.class.opcode != Op::TypeInt {
        return false;
    }

    inst.operands
        .get(1)
        .and_then(|op| match op {
            Operand::LiteralBit32(s) => Some(*s != 0),
            _ => None,
        })
        .unwrap_or(false)
}

/// Extract the bit width from an OpTypeInt or OpTypeFloat instruction.
fn get_type_bit_width(inst: &rspirv::dr::Instruction) -> Option<u32> {
    match inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => inst.operands.first().and_then(|op| match op {
            Operand::LiteralBit32(w) => Some(*w),
            _ => None,
        }),
        _ => None,
    }
}

// ============================================================================
// Constant Literal Rule
// ============================================================================

/// Validates that constant literals have correctly encoded upper bits.
///
/// This rule checks OpConstant and OpSpecConstant instructions to ensure
/// that the literal values have properly sign/zero extended upper bits
/// for integer types with bit widths less than 32.
pub struct ConstantLiteralEncodingRule;

impl ValidationRule for ConstantLiteralEncodingRule {
    fn name(&self) -> &'static str {
        "constant-literal-encoding"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Process all OpConstant and OpSpecConstant instructions
        for inst in ctx.module.types_global_values.iter() {
            let op = inst.class.opcode;
            if op != Op::Constant && op != Op::SpecConstant {
                continue;
            }

            // Get the result type
            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            // Look up the type instruction
            let type_inst = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            let Some(type_inst) = type_inst else {
                continue;
            };

            // Only validate integer types - floats use IEEE encoding
            // and the "upper bits" interpretation differs
            if type_inst.class.opcode != Op::TypeInt {
                continue;
            }

            // Get bit width and signedness
            let Some(bit_width) = get_type_bit_width(type_inst) else {
                continue;
            };

            // Only check types with bit width < 32
            if bit_width >= 32 {
                continue;
            }

            let is_signed = get_int_signedness(type_inst);
            let kind = if is_signed {
                NumberKind::SignedInt
            } else {
                NumberKind::UnsignedInt
            };

            // Get the literal value (first operand of OpConstant)
            let value = inst.operands.first().and_then(|op| match op {
                Operand::LiteralBit32(v) => Some(*v),
                _ => None,
            });

            let Some(value) = value else {
                continue;
            };

            // Verify the upper bits are correctly encoded
            if !verify_upper_bits(value, bit_width, kind) {
                if let (Some(result_id), Ok(type_id)) = (
                    inst.result_id.and_then(|id| Id::try_from(id).ok()),
                    TypeId::try_from(result_type_id),
                ) {
                    return Err(ValidationError::LiteralUpperBitsInvalid {
                        id: result_id,
                        type_id,
                        bit_width,
                        is_signed,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All literals rules
// ============================================================================

/// Returns all literal validation rules.
pub fn all_literal_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&ConstantLiteralEncodingRule]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // verify_upper_bits tests
    // ========================================================================

    #[test]
    fn test_verify_upper_bits_unsigned_8bit_valid() {
        // Valid 8-bit unsigned: value fits in 8 bits, upper bits are zero
        assert!(verify_upper_bits(0x00, 8, NumberKind::UnsignedInt));
        assert!(verify_upper_bits(0x7F, 8, NumberKind::UnsignedInt));
        assert!(verify_upper_bits(0xFF, 8, NumberKind::UnsignedInt));
    }

    #[test]
    fn test_verify_upper_bits_unsigned_8bit_invalid() {
        // Invalid 8-bit unsigned: upper bits are not zero
        assert!(!verify_upper_bits(0x100, 8, NumberKind::UnsignedInt));
        assert!(!verify_upper_bits(0xFFFF_FF00, 8, NumberKind::UnsignedInt));
    }

    #[test]
    fn test_verify_upper_bits_signed_8bit_positive_valid() {
        // Valid 8-bit signed positive: sign bit is 0, upper bits are zero
        assert!(verify_upper_bits(0x00, 8, NumberKind::SignedInt));
        assert!(verify_upper_bits(0x7F, 8, NumberKind::SignedInt));
    }

    #[test]
    fn test_verify_upper_bits_signed_8bit_positive_invalid() {
        // Invalid 8-bit signed positive: upper bits should be zero but aren't
        assert!(!verify_upper_bits(0xFFFF_FF7F, 8, NumberKind::SignedInt));
    }

    #[test]
    fn test_verify_upper_bits_signed_8bit_negative_valid() {
        // Valid 8-bit signed negative: sign bit is 1, upper bits are sign-extended
        // -1 as 8-bit signed = 0xFF, sign-extended to 0xFFFF_FFFF
        assert!(verify_upper_bits(0xFFFF_FFFF, 8, NumberKind::SignedInt));
        // -128 as 8-bit signed = 0x80, sign-extended to 0xFFFF_FF80
        assert!(verify_upper_bits(0xFFFF_FF80, 8, NumberKind::SignedInt));
    }

    #[test]
    fn test_verify_upper_bits_signed_8bit_negative_invalid() {
        // Invalid 8-bit signed negative: sign bit is 1, but upper bits not extended
        // 0x80 has sign bit set but upper bits are zero (not sign-extended)
        assert!(!verify_upper_bits(0x80, 8, NumberKind::SignedInt));
        assert!(!verify_upper_bits(0xFF, 8, NumberKind::SignedInt));
    }

    #[test]
    fn test_verify_upper_bits_16bit_unsigned() {
        assert!(verify_upper_bits(0x0000, 16, NumberKind::UnsignedInt));
        assert!(verify_upper_bits(0xFFFF, 16, NumberKind::UnsignedInt));
        assert!(!verify_upper_bits(0x1_0000, 16, NumberKind::UnsignedInt));
        assert!(!verify_upper_bits(0xFFFF_0000, 16, NumberKind::UnsignedInt));
    }

    #[test]
    fn test_verify_upper_bits_16bit_signed() {
        // Positive values (sign bit = 0, upper bits = 0)
        assert!(verify_upper_bits(0x0000, 16, NumberKind::SignedInt));
        assert!(verify_upper_bits(0x7FFF, 16, NumberKind::SignedInt));
        assert!(!verify_upper_bits(0xFFFF_7FFF, 16, NumberKind::SignedInt));

        // Negative values (sign bit = 1, upper bits = sign-extended)
        assert!(verify_upper_bits(0xFFFF_FFFF, 16, NumberKind::SignedInt)); // -1
        assert!(verify_upper_bits(0xFFFF_8000, 16, NumberKind::SignedInt)); // -32768
        assert!(!verify_upper_bits(0x8000, 16, NumberKind::SignedInt)); // Not sign-extended
        assert!(!verify_upper_bits(0xFFFF, 16, NumberKind::SignedInt)); // Not sign-extended
    }

    #[test]
    fn test_verify_upper_bits_32bit_always_valid() {
        // 32-bit values always pass (no upper bits to check)
        assert!(verify_upper_bits(0x0000_0000, 32, NumberKind::UnsignedInt));
        assert!(verify_upper_bits(0xFFFF_FFFF, 32, NumberKind::UnsignedInt));
        assert!(verify_upper_bits(0x0000_0000, 32, NumberKind::SignedInt));
        assert!(verify_upper_bits(0xFFFF_FFFF, 32, NumberKind::SignedInt));
    }

    #[test]
    fn test_verify_upper_bits_float() {
        // Floats should have zero upper bits (for sub-32-bit widths)
        assert!(verify_upper_bits(0x0000, 16, NumberKind::Float));
        assert!(verify_upper_bits(0x7FFF, 16, NumberKind::Float));
        assert!(!verify_upper_bits(0xFFFF_7FFF, 16, NumberKind::Float));
    }

    // ========================================================================
    // NumberKind tests
    // ========================================================================

    #[test]
    fn test_number_kind_equality() {
        assert_eq!(NumberKind::SignedInt, NumberKind::SignedInt);
        assert_eq!(NumberKind::UnsignedInt, NumberKind::UnsignedInt);
        assert_eq!(NumberKind::Float, NumberKind::Float);
        assert_ne!(NumberKind::SignedInt, NumberKind::UnsignedInt);
    }
}
