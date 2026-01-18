//! Constant instruction validation rules.
//!
//! This module validates SPIR-V constant instructions:
//!
//! - OpConstantTrue/OpConstantFalse: Boolean constant type validation
//! - OpSpecConstantTrue/OpSpecConstantFalse: Spec constant boolean type validation
//! - OpConstantComposite/OpSpecConstantComposite: Composite constant validation
//! - OpConstantSampler: Sampler constant type validation
//! - OpConstantNull: Null constant type validation
//! - OpSpecConstant: Spec constant type validation
//! - OpSpecConstantOp: Spec constant operation capability requirements
//!
//! Constant validation ensures type correctness and capability requirements.

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{get_type_structure, is_constant_opcode};
use crate::validation::types::{Id, ResultId, ScalarKind, TypeId, TypeStructure};
use crate::version::SpirvVersion;

// ============================================================================
// Helper Functions
// ============================================================================

/// Returns true if the given opcode is a constant or undef instruction.
fn is_constant_or_undef(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::ConstantTrue
            | Op::ConstantFalse
            | Op::Constant
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::ConstantNull
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
            | Op::Undef
    )
}

/// Returns true if the given opcode is a composite type (excluding shaped tensors which need special handling).
fn is_composite_type(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeArray
            | Op::TypeStruct
            | Op::TypeCooperativeMatrixNV
            | Op::TypeCooperativeMatrixKHR
            | Op::TypeCooperativeVectorNV
    )
}

/// Check if a type instruction is a shaped TensorARM (has shape operand).
/// A shaped tensor has: opcode, result_id, element_type, rank, shape (5 operands total in words).
fn is_shaped_tensor(type_inst: &rspirv::dr::Instruction) -> bool {
    type_inst.class.opcode == Op::TypeTensorARM && type_inst.operands.len() >= 3
}

/// Check if a type is nullable (can have a null value).
fn is_type_nullable(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    let type_inst = ResultId::try_from(type_id)
        .ok()
        .and_then(|rid| ctx.definitions.get(&rid));

    let Some(type_inst) = type_inst else {
        return false;
    };

    match type_inst.class.opcode {
        // Scalar types are nullable
        Op::TypeBool | Op::TypeInt | Op::TypeFloat => true,
        // Event types are nullable
        Op::TypeEvent | Op::TypeDeviceEvent | Op::TypeReserveId | Op::TypeQueue => true,
        // Composite types are nullable if their element type is nullable
        Op::TypeArray | Op::TypeMatrix | Op::TypeVector => {
            let element_type_id = type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            element_type_id
                .map(|id| is_type_nullable(id, ctx))
                .unwrap_or(false)
        }
        Op::TypeCooperativeMatrixNV
        | Op::TypeCooperativeMatrixKHR
        | Op::TypeCooperativeVectorNV => {
            let component_type_id = type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            component_type_id
                .map(|id| is_type_nullable(id, ctx))
                .unwrap_or(false)
        }
        // Struct is nullable if all members are nullable
        Op::TypeStruct => type_inst.operands.iter().all(|op| match op {
            Operand::IdRef(id) => is_type_nullable(*id, ctx),
            _ => true,
        }),
        // Pointers are nullable except for PhysicalStorageBuffer
        Op::TypePointer | Op::TypeUntypedPointerKHR => {
            let storage_class = type_inst.operands.first().and_then(|op| match op {
                Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            });
            storage_class != Some(StorageClass::PhysicalStorageBuffer)
        }
        // TensorARM is nullable if it has a shape and its element type is nullable
        Op::TypeTensorARM => {
            // Shaped tensors (with shape operand) can be null if element type is nullable
            // Unshaped tensors cannot be null
            if !is_shaped_tensor(type_inst) {
                return false;
            }
            let element_type_id = type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            element_type_id
                .map(|id| is_type_nullable(id, ctx))
                .unwrap_or(false)
        }
        _ => false,
    }
}

/// Check if a type is a pointer type.
fn is_pointer_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    matches!(
        ty,
        TypeStructure::Pointer { .. } | TypeStructure::ForwardPointer { .. }
    )
}

/// Check if a type contains a limited-use 8/16-bit int or float.
///
/// Returns true if the type contains 8-bit int (without Int8 capability),
/// 16-bit int (without Int16 capability), or 16-bit float (without Float16 capability).
fn contains_limited_use_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let has_int8 = ctx.has_capability(Capability::Int8);
    let has_int16 = ctx.has_capability(Capability::Int16);
    let has_float16 = ctx.has_capability(Capability::Float16);

    contains_limited_use_type_impl(type_id, ctx, has_int8, has_int16, has_float16)
}

fn contains_limited_use_type_impl(
    type_id: TypeId,
    ctx: &ValidationContext<'_>,
    has_int8: bool,
    has_int16: bool,
    has_float16: bool,
) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    match ty {
        TypeStructure::Scalar(kind) => match kind {
            ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w) => {
                let width = w.get();
                (width == 8 && !has_int8) || (width == 16 && !has_int16)
            }
            ScalarKind::Float(w) => {
                let width = w.get();
                width == 16 && !has_float16
            }
            _ => false,
        },
        TypeStructure::Vector { component, .. } | TypeStructure::Matrix { component, .. } => {
            match component {
                ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w) => {
                    let width = w.get();
                    (width == 8 && !has_int8) || (width == 16 && !has_int16)
                }
                ScalarKind::Float(w) => {
                    let width = w.get();
                    width == 16 && !has_float16
                }
                _ => false,
            }
        }
        TypeStructure::Array { element, .. } | TypeStructure::RuntimeArray { element } => {
            contains_limited_use_type_impl(element, ctx, has_int8, has_int16, has_float16)
        }
        TypeStructure::Struct { members } => members
            .iter()
            .any(|m| contains_limited_use_type_impl(*m, ctx, has_int8, has_int16, has_float16)),
        TypeStructure::Pointer { pointee, .. } => {
            if let Some(p) = pointee {
                contains_limited_use_type_impl(p, ctx, has_int8, has_int16, has_float16)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if the opcode is a constant instruction.
fn is_constant_op(opcode: Op) -> bool {
    is_constant_opcode(opcode)
}

/// Evaluate a value ID as a constant 32-bit unsigned integer if possible.
fn eval_const_u32(id: u32, ctx: &ValidationContext<'_>) -> Option<u32> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = ctx.definitions.get(&result_id)?;

    if !is_constant_opcode(inst.class.opcode) {
        return None;
    }

    if inst.class.opcode == Op::Constant || inst.class.opcode == Op::SpecConstant {
        if let Some(Operand::LiteralBit32(value)) = inst.operands.first() {
            return Some(*value);
        }
    }

    None
}

// ============================================================================
// Boolean Constant Type Rule
// ============================================================================

/// Validates boolean constant instructions.
///
/// Ensures that OpConstantTrue, OpConstantFalse, OpSpecConstantTrue,
/// and OpSpecConstantFalse have OpTypeBool as their result type.
pub struct ConstantBoolTypeRule;

impl ValidationRule for ConstantBoolTypeRule {
    fn name(&self) -> &'static str {
        "constant-bool-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            let opcode = inst.class.opcode;
            if !matches!(
                opcode,
                Op::ConstantTrue
                    | Op::ConstantFalse
                    | Op::SpecConstantTrue
                    | Op::SpecConstantFalse
            ) {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if type_opcode != Some(Op::TypeBool) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode,
                        result_type,
                        expected: "OpTypeBool",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Sampler Constant Type Rule
// ============================================================================

/// Validates OpConstantSampler instructions.
///
/// Ensures that the result type is OpTypeSampler.
pub struct ConstantSamplerTypeRule;

impl ValidationRule for ConstantSamplerTypeRule {
    fn name(&self) -> &'static str {
        "constant-sampler-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::ConstantSampler {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if type_opcode != Some(Op::TypeSampler) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode: Op::ConstantSampler,
                        result_type,
                        expected: "OpTypeSampler",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Null Constant Type Rule
// ============================================================================

/// Validates OpConstantNull instructions.
///
/// Ensures that the result type is a nullable type.
pub struct ConstantNullTypeRule;

impl ValidationRule for ConstantNullTypeRule {
    fn name(&self) -> &'static str {
        "constant-null-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::ConstantNull {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            if !is_type_nullable(result_type_id, ctx) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantNullTypeNotNullable { result_type });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Spec Constant Type Rule
// ============================================================================

/// Validates OpSpecConstant instructions.
///
/// Ensures that the result type is an integer or floating-point type.
pub struct SpecConstantTypeRule;

impl ValidationRule for SpecConstantTypeRule {
    fn name(&self) -> &'static str {
        "spec-constant-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::SpecConstant {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let type_opcode = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            if !matches!(type_opcode, Some(Op::TypeInt) | Some(Op::TypeFloat)) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode: Op::SpecConstant,
                        result_type,
                        expected: "OpTypeInt or OpTypeFloat",
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Composite Constant Type Rule
// ============================================================================

/// Validates OpConstantComposite and OpSpecConstantComposite instructions.
///
/// Ensures that:
/// - Result type is a composite type
/// - Constituent count matches the type's expected element count
/// - All constituents are constants or undef
/// - Constituent types match the expected element types
pub struct ConstantCompositeRule;

impl ValidationRule for ConstantCompositeRule {
    fn name(&self) -> &'static str {
        "constant-composite"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            let opcode = inst.class.opcode;
            if !matches!(opcode, Op::ConstantComposite | Op::SpecConstantComposite) {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let result_type_inst = ResultId::try_from(result_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            let Some(result_type_inst) = result_type_inst else {
                continue;
            };

            // Check that result type is a composite type (or shaped tensor)
            if !is_composite_type(result_type_inst.class.opcode) && !is_shaped_tensor(result_type_inst) {
                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                    return Err(ValidationError::ConstantResultTypeInvalid {
                        opcode,
                        result_type,
                        expected: "a composite type",
                    });
                }
            }

            // Validate constituents are constants or undef
            for constituent_op in &inst.operands {
                let constituent_id = match constituent_op {
                    Operand::IdRef(id) => *id,
                    _ => continue,
                };

                let constituent_opcode = ResultId::try_from(constituent_id)
                    .ok()
                    .and_then(|rid| ctx.opcodes.get(&rid))
                    .copied();

                if let Some(op) = constituent_opcode {
                    if !is_constant_or_undef(op) {
                        if let Ok(constituent) = Id::try_from(constituent_id) {
                            return Err(ValidationError::ConstantCompositeConstituentNotConstant {
                                opcode,
                                constituent,
                            });
                        }
                    }
                }
            }

            // Validate type-specific requirements
            match result_type_inst.class.opcode {
                Op::TypeVector => {
                    // Get expected component count
                    let component_count = result_type_inst
                        .operands
                        .get(1)
                        .and_then(|op| match op {
                            Operand::LiteralBit32(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);

                    let constituent_count = inst.operands.len();

                    if constituent_count != component_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: component_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeMatrix => {
                    // Get expected column count
                    let column_count = result_type_inst
                        .operands
                        .get(1)
                        .and_then(|op| match op {
                            Operand::LiteralBit32(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);

                    let constituent_count = inst.operands.len();

                    if constituent_count != column_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: column_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeArray => {
                    // Get array length from the Length operand (second operand)
                    let length_id = result_type_inst.operands.get(1).and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });

                    if let Some(length_id) = length_id {
                        // Try to evaluate the length as a constant
                        if let Some(array_length) = eval_const_u32(length_id, ctx) {
                            let constituent_count = inst.operands.len();
                            if constituent_count != array_length as usize {
                                if let Ok(result_type) = TypeId::try_from(result_type_id) {
                                    return Err(ValidationError::ConstantCompositeCountMismatch {
                                        opcode,
                                        result_type,
                                        expected: array_length as usize,
                                        found: constituent_count,
                                    });
                                }
                            }
                        }
                    }
                }
                Op::TypeStruct => {
                    // Struct member count
                    let member_count = result_type_inst.operands.len();
                    let constituent_count = inst.operands.len();

                    if constituent_count != member_count {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: member_count,
                                found: constituent_count,
                            });
                        }
                    }
                }
                Op::TypeCooperativeMatrixNV | Op::TypeCooperativeMatrixKHR => {
                    // Cooperative matrix requires exactly 1 constituent
                    let constituent_count = inst.operands.len();

                    if constituent_count != 1 {
                        if let Ok(result_type) = TypeId::try_from(result_type_id) {
                            return Err(ValidationError::ConstantCompositeCountMismatch {
                                opcode,
                                result_type,
                                expected: 1,
                                found: constituent_count,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ============================================================================
// Spec Constant Op Capability Rule
// ============================================================================

/// Validates OpSpecConstantOp capability requirements.
///
/// Certain operations within OpSpecConstantOp require specific capabilities.
pub struct SpecConstantOpCapabilityRule;

impl ValidationRule for SpecConstantOpCapabilityRule {
    fn name(&self) -> &'static str {
        "spec-constant-op-capability"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let has_shader = ctx.declared_capabilities.contains(&Capability::Shader);
        let has_kernel = ctx.declared_capabilities.contains(&Capability::Kernel);

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::SpecConstantOp {
                continue;
            }

            // The opcode is the first operand (encoded as a literal)
            let inner_op_num = inst.operands.first().and_then(|op| match op {
                Operand::LiteralBit32(n) => Some(*n),
                _ => None,
            });

            let Some(inner_op_num) = inner_op_num else {
                continue;
            };

            // Check capability requirements for specific ops by their opcode numbers
            // See SPIR-V specification for opcode values
            // QuantizeToF16 = 116
            if inner_op_num == 116 && !has_shader {
                return Err(ValidationError::SpecConstantOpMissingCapability {
                    inner_opcode: Op::QuantizeToF16,
                    required_capability: Capability::Shader,
                });
            }

            // Kernel-only operations:
            // ConvertFToS = 109, ConvertSToF = 111, ConvertFToU = 110, ConvertUToF = 112
            // ConvertPtrToU = 117, ConvertUToPtr = 120, GenericCastToPtr = 122, PtrCastToGeneric = 121
            // Bitcast = 124, FNegate = 127, FAdd = 129, FSub = 131, FMul = 133, FDiv = 136, FRem = 140, FMod = 141
            // AccessChain = 65, InBoundsAccessChain = 66, PtrAccessChain = 67, InBoundsPtrAccessChain = 70
            let kernel_ops = [
                109, 111, 110, 112, // ConvertFToS, ConvertSToF, ConvertFToU, ConvertUToF
                117, 120, 122, 121, // ConvertPtrToU, ConvertUToPtr, GenericCastToPtr, PtrCastToGeneric
                124, 127, 129, 131, 133, 136, 140, 141, // Bitcast, FNegate, FAdd, FSub, FMul, FDiv, FRem, FMod
                65, 66, 67, 70, // AccessChain, InBoundsAccessChain, PtrAccessChain, InBoundsPtrAccessChain
            ];

            if kernel_ops.contains(&inner_op_num) && !has_kernel {
                // Map the number back to an Op for the error message
                let inner_op = match inner_op_num {
                    109 => Op::ConvertFToS,
                    111 => Op::ConvertSToF,
                    110 => Op::ConvertFToU,
                    112 => Op::ConvertUToF,
                    117 => Op::ConvertPtrToU,
                    120 => Op::ConvertUToPtr,
                    122 => Op::GenericCastToPtr,
                    121 => Op::PtrCastToGeneric,
                    124 => Op::Bitcast,
                    127 => Op::FNegate,
                    129 => Op::FAdd,
                    131 => Op::FSub,
                    133 => Op::FMul,
                    136 => Op::FDiv,
                    140 => Op::FRem,
                    141 => Op::FMod,
                    65 => Op::AccessChain,
                    66 => Op::InBoundsAccessChain,
                    67 => Op::PtrAccessChain,
                    70 => Op::InBoundsPtrAccessChain,
                    _ => continue,
                };
                return Err(ValidationError::SpecConstantOpMissingCapability {
                    inner_opcode: inner_op,
                    required_capability: Capability::Kernel,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Small Type Constant Rule
// ============================================================================

/// Validates that 8- or 16-bit constants are not formed without full capabilities.
///
/// When the Shader capability is present but full 8/16-bit capabilities are not,
/// creating constants of these types is disallowed.
pub struct SmallTypeConstantRule;

impl ValidationRule for SmallTypeConstantRule {
    fn name(&self) -> &'static str {
        "small-type-constant"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Only applies when Shader capability is present
        if !ctx.has_capability(Capability::Shader) {
            return Ok(());
        }

        for inst in &ctx.module.types_global_values {
            // Check constant instructions
            if !is_constant_op(inst.class.opcode) {
                continue;
            }

            let Some(result_type_id) = inst.result_type else {
                continue;
            };

            let Ok(type_id) = TypeId::try_from(result_type_id) else {
                continue;
            };

            // Skip pointer types
            if is_pointer_type(type_id, ctx) {
                continue;
            }

            // Check for limited-use small types
            if contains_limited_use_type(type_id, ctx) {
                return Err(ValidationError::ConstantSmallTypeNotAllowed);
            }
        }

        Ok(())
    }
}

// ============================================================================
// UConvert Spec Constant Op Rule
// ============================================================================

/// Validates OpSpecConstantOp UConvert requirements before SPIR-V 1.4.
///
/// Prior to SPIR-V 1.4, OpSpecConstantOp with UConvert requires:
/// - Kernel capability, OR
/// - SPV_AMD_gpu_shader_int16 extension
pub struct SpecConstantOpUConvertRule;

impl ValidationRule for SpecConstantOpUConvertRule {
    fn name(&self) -> &'static str {
        "spec-constant-op-uconvert"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Only check before SPIR-V 1.4
        let v1_4 = SpirvVersion::new(1, 4);
        if ctx.target_version >= v1_4 {
            return Ok(());
        }

        // UConvert is allowed with Kernel capability
        if ctx.has_capability(Capability::Kernel) {
            return Ok(());
        }

        // Check if SPV_AMD_gpu_shader_int16 extension is present
        let has_amd_ext = ctx.module.extensions.iter().any(|ext| {
            ext.operands.first().map_or(false, |op| {
                matches!(op, Operand::LiteralString(s) if s == "SPV_AMD_gpu_shader_int16")
            })
        });

        if has_amd_ext {
            return Ok(());
        }

        // Check for OpSpecConstantOp with UConvert
        // UConvert opcode number is 113
        const UCONVERT_OPCODE: u32 = 113;

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::SpecConstantOp {
                continue;
            }

            let inner_op_num = inst.operands.first().and_then(|op| match op {
                Operand::LiteralBit32(n) => Some(*n),
                _ => None,
            });

            if inner_op_num == Some(UCONVERT_OPCODE) {
                return Err(ValidationError::SpecConstantOpUConvertRequiresKernel);
            }
        }

        Ok(())
    }
}

// ============================================================================
// All constant rules
// ============================================================================

/// Returns all constant validation rules.
pub fn all_constant_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ConstantBoolTypeRule,
        &ConstantSamplerTypeRule,
        &ConstantNullTypeRule,
        &SpecConstantTypeRule,
        &ConstantCompositeRule,
        &SpecConstantOpCapabilityRule,
        &SmallTypeConstantRule,
        &SpecConstantOpUConvertRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_constant_or_undef() {
        assert!(is_constant_or_undef(Op::ConstantTrue));
        assert!(is_constant_or_undef(Op::ConstantFalse));
        assert!(is_constant_or_undef(Op::Constant));
        assert!(is_constant_or_undef(Op::ConstantComposite));
        assert!(is_constant_or_undef(Op::ConstantNull));
        assert!(is_constant_or_undef(Op::Undef));
        assert!(is_constant_or_undef(Op::SpecConstant));
        assert!(is_constant_or_undef(Op::SpecConstantOp));

        assert!(!is_constant_or_undef(Op::IAdd));
        assert!(!is_constant_or_undef(Op::Variable));
        assert!(!is_constant_or_undef(Op::Load));
    }

    #[test]
    fn test_is_composite_type() {
        assert!(is_composite_type(Op::TypeVector));
        assert!(is_composite_type(Op::TypeMatrix));
        assert!(is_composite_type(Op::TypeArray));
        assert!(is_composite_type(Op::TypeStruct));
        assert!(is_composite_type(Op::TypeCooperativeMatrixNV));
        assert!(is_composite_type(Op::TypeCooperativeMatrixKHR));
        assert!(is_composite_type(Op::TypeCooperativeVectorNV));

        assert!(!is_composite_type(Op::TypeInt));
        assert!(!is_composite_type(Op::TypeFloat));
        assert!(!is_composite_type(Op::TypeBool));
        assert!(!is_composite_type(Op::TypePointer));
    }
}
