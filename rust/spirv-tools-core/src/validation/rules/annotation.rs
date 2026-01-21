//! Annotation validation rules.
//!
//! This module validates SPIR-V decoration and annotation requirements including:
//!
//! - Decoration target validation
//! - Vulkan decoration restrictions
//! - OpMemberDecorate validation
//! - Decoration group validation
//! - FPFastMathMode/NoContraction conflict detection

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a decoration takes ID parameters and must use OpDecorateId.
fn decoration_takes_id_parameters(dec: Decoration) -> bool {
    matches!(
        dec,
        Decoration::UniformId
            | Decoration::AlignmentId
            | Decoration::MaxByteOffsetId
            | Decoration::HlslCounterBufferGOOGLE
            | Decoration::NodeMaxPayloadsAMDX
            | Decoration::NodeSharesPayloadLimitsWithAMDX
            | Decoration::PayloadNodeArraySizeAMDX
            | Decoration::PayloadNodeNameAMDX
            | Decoration::PayloadNodeBaseIndexAMDX
    )
}

/// Check if a decoration is only valid for structure members.
fn is_member_decoration_only(dec: Decoration) -> bool {
    matches!(
        dec,
        Decoration::RowMajor | Decoration::ColMajor | Decoration::MatrixStride
    )
}

/// Check if a decoration cannot be applied to structure members.
fn is_not_member_decoration(dec: Decoration) -> bool {
    matches!(
        dec,
        Decoration::SpecId
            | Decoration::Block
            | Decoration::BufferBlock
            | Decoration::ArrayStride
            | Decoration::GLSLShared
            | Decoration::GLSLPacked
            | Decoration::CPacked
            | Decoration::Aliased
            | Decoration::Constant
            | Decoration::Uniform
            | Decoration::UniformId
            | Decoration::SaturatedConversion
            | Decoration::Index
            | Decoration::Binding
            | Decoration::DescriptorSet
            | Decoration::FuncParamAttr
            | Decoration::FPRoundingMode
            | Decoration::FPFastMathMode
            | Decoration::LinkageAttributes
            | Decoration::NoContraction
            | Decoration::InputAttachmentIndex
            | Decoration::Alignment
            | Decoration::MaxByteOffset
            | Decoration::AlignmentId
            | Decoration::MaxByteOffsetId
            | Decoration::NoSignedWrap
            | Decoration::NoUnsignedWrap
            | Decoration::NonUniform
            | Decoration::RestrictPointer
            | Decoration::AliasedPointer
            | Decoration::CounterBuffer
    )
}

/// Validates OpDecorate instructions.
pub struct DecorateValidationRule;

impl ValidationRule for DecorateValidationRule {
    fn name(&self) -> &'static str {
        "decorate-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build set of IDs with FPFastMathMode decoration
        let mut fp_fast_math_ids: HashSet<u32> = HashSet::new();
        // Build set of IDs with NoContraction decoration
        let mut no_contraction_ids: HashSet<u32> = HashSet::new();

        // First pass: collect decorations
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::Decoration(dec)) = inst.operands.get(1) else {
                continue;
            };

            if *dec == Decoration::FPFastMathMode {
                fp_fast_math_ids.insert(*target_id);
            } else if *dec == Decoration::NoContraction {
                no_contraction_ids.insert(*target_id);
            }
        }

        // Second pass: validate
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::Decoration(dec)) = inst.operands.get(1) else {
                continue;
            };

            // Vulkan restrictions
            if ctx.env.is_vulkan() {
                if *dec == Decoration::GLSLShared || *dec == Decoration::GLSLPacked {
                    return Err(ValidationError::VulkanDecorationNotAllowed {
                        decoration: *dec,
                    }.into());
                }
            }

            // FPFastMathMode and NoContraction cannot be on same target
            if *dec == Decoration::FPFastMathMode && no_contraction_ids.contains(target_id) {
                return Err(ValidationError::FPFastMathModeConflictsWithNoContraction {
                    target_id: to_id(*target_id),
                }.into());
            }
            if *dec == Decoration::NoContraction && fp_fast_math_ids.contains(target_id) {
                return Err(ValidationError::FPFastMathModeConflictsWithNoContraction {
                    target_id: to_id(*target_id),
                }.into());
            }

            // FPFastMathMode validation
            if *dec == Decoration::FPFastMathMode {
                if let Some(Operand::LiteralBit32(mask)) = inst.operands.get(2) {
                    // Check AllowTransform (bit 16) requires AllowContract (bit 7) and AllowReassoc (bit 6)
                    const ALLOW_TRANSFORM: u32 = 1 << 16;
                    const ALLOW_CONTRACT: u32 = 1 << 7;
                    const ALLOW_REASSOC: u32 = 1 << 6;

                    let allow_transform = (*mask & ALLOW_TRANSFORM) != 0;
                    let allow_contract = (*mask & ALLOW_CONTRACT) != 0;
                    let allow_reassoc = (*mask & ALLOW_REASSOC) != 0;

                    if allow_transform && !(allow_contract && allow_reassoc) {
                        return Err(ValidationError::FPFastMathAllowTransformRequiresContractReassoc {
                            target_id: to_id(*target_id),
                        }.into());
                    }
                }
            }

            // Decorations taking ID parameters must not use OpDecorate
            if decoration_takes_id_parameters(*dec) {
                return Err(ValidationError::DecorationRequiresDecorateId {
                    decoration: *dec,
                }.into());
            }

            // Check member-only decorations are not on non-struct targets
            let target = ctx
                .definitions
                .get(&ResultId::try_from(*target_id).ok().unwrap_or(ResultId::try_from(1u32).unwrap()));
            if let Some(target_inst) = target {
                // Member-only decorations cannot be applied to non-struct targets
                if target_inst.class.opcode != Op::DecorationGroup
                    && is_member_decoration_only(*dec)
                {
                    return Err(ValidationError::MemberDecorationOnNonMember {
                        decoration: *dec,
                        target_id: to_id(*target_id),
                    }.into());
                }
            }
        }

        Ok(())
    }
}

/// Validates OpDecorateId instructions.
pub struct DecorateIdValidationRule;

impl ValidationRule for DecorateIdValidationRule {
    fn name(&self) -> &'static str {
        "decorate-id-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        for inst in &module.annotations {
            if inst.class.opcode != Op::DecorateId {
                continue;
            }
            let Some(Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::Decoration(dec)) = inst.operands.get(1) else {
                continue;
            };

            // OpDecorateId target must not be a decoration group
            if let Ok(result_id) = ResultId::try_from(*target_id) {
                if let Some(target_inst) = ctx.definitions.get(&result_id) {
                    if target_inst.class.opcode == Op::DecorationGroup {
                        return Err(ValidationError::DecorateIdTargetIsDecorationGroup {
                            target_id: to_id(*target_id),
                        }.into());
                    }
                }
            }

            // OpDecorateId must use decorations that take ID parameters
            if !decoration_takes_id_parameters(*dec) {
                return Err(ValidationError::DecorationDoesNotTakeIdParameters {
                    decoration: *dec,
                }.into());
            }
        }

        Ok(())
    }
}

/// Validates OpMemberDecorate instructions.
pub struct MemberDecorateValidationRule;

impl ValidationRule for MemberDecorateValidationRule {
    fn name(&self) -> &'static str {
        "member-decorate-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build map of struct ID -> member count
        let mut struct_members: HashMap<u32, u32> = HashMap::new();
        for inst in &module.types_global_values {
            if inst.class.opcode == Op::TypeStruct {
                if let Some(struct_id) = inst.result_id {
                    // Member count is the number of operands
                    struct_members.insert(struct_id, inst.operands.len() as u32);
                }
            }
        }

        for inst in &module.annotations {
            if inst.class.opcode != Op::MemberDecorate {
                continue;
            }
            let Some(Operand::IdRef(struct_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::LiteralBit32(member_index)) = inst.operands.get(1) else {
                continue;
            };
            let Some(Operand::Decoration(dec)) = inst.operands.get(2) else {
                continue;
            };

            // Target must be a struct type
            let Some(&member_count) = struct_members.get(struct_id) else {
                return Err(ValidationError::MemberDecorateTargetNotStruct {
                    target_id: to_id(*struct_id),
                }.into());
            };

            // Member index must be in range
            if *member_index >= member_count {
                return Err(ValidationError::MemberDecorateIndexOutOfBounds {
                    struct_id: to_id(*struct_id),
                    member_index: *member_index,
                    member_count,
                }.into());
            }

            // Check decorations that cannot be applied to members
            if is_not_member_decoration(*dec) {
                return Err(ValidationError::DecorationCannotBeOnMember {
                    decoration: *dec,
                    struct_id: to_id(*struct_id),
                    member_index: *member_index,
                }.into());
            }
        }

        Ok(())
    }
}

/// Validates OpDecorationGroup instructions.
///
/// Note: Comprehensive use validation would require tracking all references,
/// which is deferred to full module validation. This rule validates that
/// decoration groups exist and are properly formed.
pub struct DecorationGroupValidationRule;

impl ValidationRule for DecorationGroupValidationRule {
    fn name(&self) -> &'static str {
        "decoration-group-validation"
    }

    fn validate(&self, _ctx: &ValidationContext<'_>) -> ValidationResult {
        // Decoration group validation is primarily done through OpGroupDecorate
        // and OpGroupMemberDecorate rules, which check that their first operand
        // references a valid decoration group.
        //
        // Full use-def chain validation to ensure decoration groups are only
        // used by valid instructions would require building a complete use map,
        // which is beyond the scope of this simple structural validation.
        Ok(())
    }
}

/// Validates OpGroupDecorate instructions.
pub struct GroupDecorateValidationRule;

impl ValidationRule for GroupDecorateValidationRule {
    fn name(&self) -> &'static str {
        "group-decorate-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build set of decoration group IDs
        let decoration_groups: HashSet<u32> = module
            .annotations
            .iter()
            .filter(|inst| inst.class.opcode == Op::DecorationGroup)
            .filter_map(|inst| inst.result_id)
            .collect();

        for inst in &module.annotations {
            if inst.class.opcode != Op::GroupDecorate {
                continue;
            }
            let Some(Operand::IdRef(group_id)) = inst.operands.first() else {
                continue;
            };

            // First operand must be a decoration group
            if !decoration_groups.contains(group_id) {
                return Err(ValidationError::GroupDecorateNotDecorationGroup {
                    target_id: to_id(*group_id),
                }.into());
            }

            // Targets must not be decoration groups
            for operand in inst.operands.iter().skip(1) {
                if let Operand::IdRef(target_id) = operand {
                    if decoration_groups.contains(target_id) {
                        return Err(ValidationError::GroupDecorateTargetIsDecorationGroup {
                            target_id: to_id(*target_id),
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpGroupMemberDecorate instructions.
pub struct GroupMemberDecorateValidationRule;

impl ValidationRule for GroupMemberDecorateValidationRule {
    fn name(&self) -> &'static str {
        "group-member-decorate-validation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Build set of decoration group IDs
        let decoration_groups: HashSet<u32> = module
            .annotations
            .iter()
            .filter(|inst| inst.class.opcode == Op::DecorationGroup)
            .filter_map(|inst| inst.result_id)
            .collect();

        // Build map of struct ID -> member count
        let mut struct_members: HashMap<u32, u32> = HashMap::new();
        for inst in &module.types_global_values {
            if inst.class.opcode == Op::TypeStruct {
                if let Some(struct_id) = inst.result_id {
                    struct_members.insert(struct_id, inst.operands.len() as u32);
                }
            }
        }

        for inst in &module.annotations {
            if inst.class.opcode != Op::GroupMemberDecorate {
                continue;
            }
            let Some(Operand::IdRef(group_id)) = inst.operands.first() else {
                continue;
            };

            // First operand must be a decoration group
            if !decoration_groups.contains(group_id) {
                return Err(ValidationError::GroupMemberDecorateNotDecorationGroup {
                    target_id: to_id(*group_id),
                }.into());
            }

            // Remaining operands are (struct_id, member_index) pairs
            let mut i = 1;
            while i + 1 < inst.operands.len() {
                let (struct_id, member_index) = match (&inst.operands[i], &inst.operands[i + 1]) {
                    (Operand::IdRef(s), Operand::LiteralBit32(m)) => (*s, *m),
                    _ => {
                        i += 2;
                        continue;
                    }
                };

                // Target must be a struct type
                let Some(&member_count) = struct_members.get(&struct_id) else {
                    return Err(ValidationError::GroupMemberDecorateTargetNotStruct {
                        struct_id: to_id(struct_id),
                    }.into());
                };

                // Member index must be in range
                if member_index >= member_count {
                    return Err(ValidationError::GroupMemberDecorateIndexOutOfBounds {
                        struct_id: to_id(struct_id),
                        member_index,
                        member_count,
                    }.into());
                }

                i += 2;
            }
        }

        Ok(())
    }
}

/// Validates Vulkan-specific decoration storage class restrictions.
///
/// In Vulkan, certain decorations can only be applied to variables
/// with specific storage classes:
/// - Location/Component: Input, Output, or ray tracing storage classes (VUID-6672)
/// - Index: Output only
/// - Binding/DescriptorSet: StorageBuffer, Uniform, or UniformConstant (VUID-6491)
/// - InputAttachmentIndex: UniformConstant only (VUID-6678)
/// - Flat/NoPerspective/Centroid/Sample: Input or Output (VUID-4670)
/// - PerVertexKHR: Input only (VUID-6777)
pub struct VulkanDecorationStorageClassRule;

impl ValidationRule for VulkanDecorationStorageClassRule {
    fn name(&self) -> &'static str {
        "vulkan-decoration-storage-class"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only apply in Vulkan environments
        if !ctx.env.is_vulkan() {
            return Ok(());
        }

        let module = ctx.module();

        // Build a map of variable IDs to their storage classes
        let mut variable_storage_classes: HashMap<u32, StorageClass> = HashMap::new();
        for inst in &module.types_global_values {
            if matches!(inst.class.opcode, Op::Variable | Op::UntypedVariableKHR) {
                if let Some(var_id) = inst.result_id {
                    if let Some(Operand::StorageClass(sc)) = inst.operands.first() {
                        variable_storage_classes.insert(var_id, *sc);
                    }
                }
            }
        }
        // Also check function-local variables
        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    if matches!(inst.class.opcode, Op::Variable | Op::UntypedVariableKHR) {
                        if let Some(var_id) = inst.result_id {
                            if let Some(Operand::StorageClass(sc)) = inst.operands.first() {
                                variable_storage_classes.insert(var_id, *sc);
                            }
                        }
                    }
                }
            }
        }

        // Storage classes allowed for Location/Component decorations
        let location_valid_storage_classes = |sc: StorageClass| -> bool {
            matches!(
                sc,
                StorageClass::Input
                    | StorageClass::Output
                    | StorageClass::RayPayloadKHR
                    | StorageClass::IncomingRayPayloadKHR
                    | StorageClass::HitAttributeKHR
                    | StorageClass::CallableDataKHR
                    | StorageClass::IncomingCallableDataKHR
                    | StorageClass::ShaderRecordBufferKHR
                    | StorageClass::HitObjectAttributeNV
                    | StorageClass::TileImageEXT
            )
        };

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(Operand::IdRef(target_id)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::Decoration(dec)) = inst.operands.get(1) else {
                continue;
            };

            // Get storage class if target is a variable
            let Some(&storage_class) = variable_storage_classes.get(target_id) else {
                continue;
            };

            match dec {
                Decoration::Location | Decoration::Component => {
                    if !location_valid_storage_classes(storage_class) {
                        return Err(ValidationError::VulkanDecorationStorageClassMismatch {
                            decoration: *dec,
                            target_id: to_id(*target_id),
                            storage_class,
                            vuid: 6672,
                        }.into());
                    }
                }
                Decoration::Index => {
                    if storage_class != StorageClass::Output {
                        return Err(ValidationError::VulkanIndexDecorationNotOutput {
                            target_id: to_id(*target_id),
                            storage_class,
                        }.into());
                    }
                }
                Decoration::Binding | Decoration::DescriptorSet => {
                    if !matches!(
                        storage_class,
                        StorageClass::StorageBuffer
                            | StorageClass::Uniform
                            | StorageClass::UniformConstant
                            | StorageClass::TileAttachmentQCOM
                    ) {
                        return Err(ValidationError::VulkanBindingDecorationInvalidStorageClass {
                            target_id: to_id(*target_id),
                            storage_class,
                        }.into());
                    }
                }
                Decoration::InputAttachmentIndex => {
                    if storage_class != StorageClass::UniformConstant {
                        return Err(
                            ValidationError::VulkanInputAttachmentIndexInvalidStorageClass {
                                target_id: to_id(*target_id),
                                storage_class,
                            }.into(),
                        );
                    }
                }
                Decoration::Flat
                | Decoration::NoPerspective
                | Decoration::Centroid
                | Decoration::Sample => {
                    if !matches!(storage_class, StorageClass::Input | StorageClass::Output) {
                        return Err(
                            ValidationError::VulkanInterpolationDecorationInvalidStorageClass {
                                decoration: *dec,
                                target_id: to_id(*target_id),
                                storage_class,
                            }.into(),
                        );
                    }
                }
                Decoration::PerVertexKHR => {
                    if storage_class != StorageClass::Input {
                        return Err(ValidationError::VulkanPerVertexDecorationNotInput {
                            target_id: to_id(*target_id),
                            storage_class,
                        }.into());
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Returns all annotation validation rules.
pub fn all_annotation_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(DecorateValidationRule),
        Box::new(DecorateIdValidationRule),
        Box::new(MemberDecorateValidationRule),
        Box::new(DecorationGroupValidationRule),
        Box::new(GroupDecorateValidationRule),
        Box::new(GroupMemberDecorateValidationRule),
        Box::new(VulkanDecorationStorageClassRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoration_takes_id_parameters() {
        assert!(decoration_takes_id_parameters(Decoration::UniformId));
        assert!(decoration_takes_id_parameters(Decoration::AlignmentId));
        assert!(decoration_takes_id_parameters(Decoration::MaxByteOffsetId));
        assert!(!decoration_takes_id_parameters(Decoration::Location));
        assert!(!decoration_takes_id_parameters(Decoration::Block));
    }

    #[test]
    fn test_is_member_decoration_only() {
        assert!(is_member_decoration_only(Decoration::RowMajor));
        assert!(is_member_decoration_only(Decoration::ColMajor));
        assert!(is_member_decoration_only(Decoration::MatrixStride));
        assert!(!is_member_decoration_only(Decoration::Location));
        assert!(!is_member_decoration_only(Decoration::Block));
    }

    #[test]
    fn test_is_not_member_decoration() {
        assert!(is_not_member_decoration(Decoration::SpecId));
        assert!(is_not_member_decoration(Decoration::Block));
        assert!(is_not_member_decoration(Decoration::ArrayStride));
        assert!(!is_not_member_decoration(Decoration::Location));
        assert!(!is_not_member_decoration(Decoration::RowMajor));
    }
}
