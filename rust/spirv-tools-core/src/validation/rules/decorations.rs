//! Decoration validation rules.
//!
//! This module validates SPIR-V decoration requirements including:
//!
//! - Decoration version requirements
//! - Decoration target categories
//! - Decoration groups
//! - Member-only decoration restrictions
//!
//! # Adding New Decoration Rules
//!
//! Decoration validation rules follow the [`ValidationRule`] trait pattern:
//!
//! ```ignore
//! pub struct MyDecorationRule;
//!
//! impl ValidationRule for MyDecorationRule {
//!     fn name(&self) -> &'static str {
//!         "my-decoration-rule"
//!     }
//!
//!     fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
//!         for inst in &ctx.module().annotations {
//!             if inst.class.opcode == Op::Decorate {
//!                 // Validate decoration...
//!             }
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use std::collections::HashSet;

use rspirv::spirv::{BuiltIn, Capability, Decoration, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_memory_object_declaration;
use crate::validation::types::{
    DecorationTargetId, DecorationTargetKind, Id, IdKind, MemberDecorationTargetId, MemberIndex,
    OperandId, ResultId,
};
use crate::version::SpirvVersion;

// ============================================================================
// Decoration Version Rules
// ============================================================================

/// Validates that decorations are compatible with the target SPIR-V version.
///
/// Some decorations (like BufferBlock) were deprecated in later SPIR-V versions.
pub struct DecorationVersionRule;

impl ValidationRule for DecorationVersionRule {
    fn name(&self) -> &'static str {
        "decoration-versions"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let target_version = ctx.target_version;

        for inst in &ctx.module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }

            if let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) {
                // BufferBlock was deprecated after SPIR-V 1.3
                if *decoration == Decoration::BufferBlock
                    && target_version > SpirvVersion::new(1, 3)
                {
                    return Err(ValidationError::DecorationRequiresSpirvVersion {
                        decoration: *decoration,
                        required_version: SpirvVersion::new(1, 3),
                        target_version,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Decoration Groups Rule
// ============================================================================

/// Validates decoration groups and their references.
pub struct DecorationGroupsRule;

impl ValidationRule for DecorationGroupsRule {
    fn name(&self) -> &'static str {
        "decoration-groups"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        let groups: HashSet<ResultId> = module
            .annotations
            .iter()
            .filter_map(|inst| {
                if inst.class.opcode == Op::DecorationGroup {
                    inst.result_id.and_then(|id| ResultId::try_from(id).ok())
                } else {
                    None
                }
            })
            .collect();

        for inst in &module.annotations {
            match inst.class.opcode {
                Op::GroupDecorate | Op::GroupMemberDecorate => {
                    let group = inst.operands.iter().find_map(|op| {
                        if let rspirv::dr::Operand::IdRef(id) = op {
                            ResultId::try_from(*id).ok()
                        } else {
                            None
                        }
                    });
                    if let Some(group) = group {
                        if !groups.contains(&group) {
                            return Err(ValidationError::UnknownDecorationGroup {
                                group: group.into_inner(),
                            });
                        }
                    }
                    let mut operands = inst.operands.iter().skip(1);
                    while let Some(operand) = operands.next() {
                        if let rspirv::dr::Operand::IdRef(id) = operand {
                            let target =
                                ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                                    kind: IdKind::Operand,
                                    opcode: inst.class.opcode,
                                })?;
                            if !ctx.defined_result_ids.contains(&target) {
                                return Err(ValidationError::MissingDecorationTarget {
                                    target: target.into_inner(),
                                });
                            }
                            if inst.class.opcode == Op::GroupMemberDecorate {
                                let member_index = operands
                                    .next()
                                    .and_then(|op| {
                                        if let rspirv::dr::Operand::LiteralBit32(member) = op {
                                            Some(*member)
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(0);
                                if let Some(opcode) = ctx.opcodes.get(&target) {
                                    if *opcode != Op::TypeStruct {
                                        let target_operand =
                                            OperandId::try_from(u32::from(target))
                                                .expect("validated non-zero id");
                                        return Err(
                                            ValidationError::MemberDecorationTargetNotStruct {
                                                target: MemberDecorationTargetId::new(
                                                    DecorationTargetId::new(target_operand),
                                                    MemberIndex::new(member_index),
                                                ),
                                            },
                                        );
                                    }
                                }
                                if let Some(member_count) = ctx.struct_member_counts.get(&target) {
                                    if (member_index as usize) >= *member_count {
                                        return Err(
                                            ValidationError::MemberDecorationIndexOutOfRange {
                                                target: DecorationTargetId::new(
                                                    OperandId::try_from(u32::from(target)).unwrap(),
                                                ),
                                                member: MemberIndex::new(member_index),
                                                member_count: *member_count,
                                            },
                                        );
                                    }
                                }
                            }
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
// Basic Decoration Validation Rule
// ============================================================================

/// Validates basic decoration requirements (targets exist, member-only decorations).
pub struct BasicDecorationRule;

impl ValidationRule for BasicDecorationRule {
    fn name(&self) -> &'static str {
        "basic-decorations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        for inst in &module.annotations {
            match inst.class.opcode {
                Op::Decorate | Op::DecorateId => {
                    if let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1)
                    {
                        if matches!(
                            decoration,
                            Decoration::Offset
                                | Decoration::MatrixStride
                                | Decoration::RowMajor
                                | Decoration::ColMajor
                        ) {
                            return Err(ValidationError::MemberOnlyDecorationUsedWithDecorate {
                                decoration: *decoration,
                            });
                        }
                    }
                    let mut operands = inst.operands.iter();
                    let target = operands.find_map(|op| {
                        if let rspirv::dr::Operand::IdRef(id) = op {
                            ResultId::try_from(*id).ok()
                        } else {
                            None
                        }
                    });
                    if let Some(target) = target {
                        if !ctx.defined_result_ids.contains(&target) {
                            return Err(ValidationError::MissingDecorationTarget {
                                target: target.into_inner(),
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
// Decoration Target Categories Rule
// ============================================================================

/// Validates that decorations are applied to appropriate target categories.
pub struct DecorationTargetCategoriesRule;

impl ValidationRule for DecorationTargetCategoriesRule {
    fn name(&self) -> &'static str {
        "decoration-target-categories"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        for inst in &module.annotations {
            if !matches!(inst.class.opcode, Op::Decorate | Op::DecorateId) {
                continue;
            }
            let mut operands = inst.operands.iter();
            let target = match operands.next() {
                Some(rspirv::dr::Operand::IdRef(id)) => {
                    ResultId::try_from(*id).map_err(|_| ValidationError::ZeroId {
                        kind: IdKind::Operand,
                        opcode: inst.class.opcode,
                    })?
                }
                _ => continue,
            };
            let decoration = match operands.next() {
                Some(rspirv::dr::Operand::Decoration(dec)) => *dec,
                _ => continue,
            };
            let target_inst = match ctx.definitions.get(&target) {
                Some(inst) => inst,
                None => continue,
            };
            let opcode = match ctx.opcodes.get(&target) {
                Some(opcode) => *opcode,
                None => continue,
            };
            let target_id = Id::try_from(u32::from(target)).expect("non-zero id validated");
            let target_type_id = target_inst.result_type;

            let expected = match decoration {
                Decoration::SpecId => {
                    if !is_scalar_spec_constant(opcode) {
                        Some(DecorationTargetKind::ScalarSpecConstant)
                    } else {
                        None
                    }
                }
                Decoration::Block
                | Decoration::BufferBlock
                | Decoration::GLSLShared
                | Decoration::GLSLPacked
                | Decoration::CPacked => {
                    if opcode != Op::TypeStruct {
                        Some(DecorationTargetKind::StructType)
                    } else {
                        None
                    }
                }
                Decoration::ArrayStride => {
                    if matches!(
                        opcode,
                        Op::TypeArray
                            | Op::TypeRuntimeArray
                            | Op::TypePointer
                            | Op::TypeUntypedPointerKHR
                    ) {
                        None
                    } else {
                        Some(DecorationTargetKind::ArrayOrPointerType)
                    }
                }
                Decoration::BuiltIn => {
                    let builtin = operands.next().and_then(|op| {
                        if let rspirv::dr::Operand::BuiltIn(value) = op {
                            Some(*value)
                        } else if let rspirv::dr::Operand::LiteralBit32(raw) = op {
                            BuiltIn::from_u32(*raw)
                        } else {
                            None
                        }
                    });
                    if ctx.declared_capabilities.contains(&Capability::Shader)
                        && builtin == Some(BuiltIn::WorkgroupSize)
                        && !is_constant_opcode(opcode)
                    {
                        Some(DecorationTargetKind::Constant)
                    } else if matches!(opcode, Op::Variable | Op::UntypedVariableKHR)
                        || is_constant_opcode(opcode)
                    {
                        None
                    } else {
                        Some(DecorationTargetKind::Variable)
                    }
                }
                Decoration::NoPerspective
                | Decoration::Flat
                | Decoration::Patch
                | Decoration::Centroid
                | Decoration::Sample
                | Decoration::Restrict
                | Decoration::Aliased
                | Decoration::Volatile
                | Decoration::Coherent
                | Decoration::NonWritable
                | Decoration::NonReadable
                | Decoration::XfbBuffer
                | Decoration::XfbStride
                | Decoration::Component
                | Decoration::Stream
                | Decoration::RestrictPointer
                | Decoration::AliasedPointer
                | Decoration::PerPrimitiveEXT => {
                    if !is_memory_object_declaration(opcode) {
                        Some(DecorationTargetKind::MemoryObjectDeclaration)
                    } else if !is_pointer_type(target_type_id, ctx.definitions) {
                        Some(DecorationTargetKind::Pointer)
                    } else {
                        None
                    }
                }
                Decoration::Invariant
                | Decoration::Constant
                | Decoration::Location
                | Decoration::Index
                | Decoration::Binding
                | Decoration::DescriptorSet
                | Decoration::InputAttachmentIndex => {
                    if matches!(opcode, Op::Variable | Op::UntypedVariableKHR) {
                        None
                    } else {
                        Some(DecorationTargetKind::Variable)
                    }
                }
                Decoration::LinkageAttributes => {
                    if matches!(opcode, Op::Function | Op::Variable | Op::UntypedVariableKHR) {
                        None
                    } else {
                        Some(DecorationTargetKind::FunctionOrVariable)
                    }
                }
                _ => None,
            };

            if let Some(expected) = expected {
                return Err(ValidationError::InvalidDecorationTargetKind {
                    decoration,
                    target: target_id,
                    found: opcode,
                    expected,
                });
            }
        }
        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn is_scalar_spec_constant(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::SpecConstantTrue | Op::SpecConstantFalse | Op::SpecConstant
    )
}

fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
    )
}

fn is_pointer_type(
    type_id: Option<u32>,
    definitions: &std::collections::HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    type_id
        .and_then(|id| ResultId::try_from(id).ok())
        .and_then(|id| definitions.get(&id))
        .map(|inst| matches!(inst.class.opcode, Op::TypePointer | Op::TypeUntypedPointerKHR))
        .unwrap_or(false)
}

// ============================================================================
// All decoration rules
// ============================================================================

/// Returns all decoration validation rules.
pub fn all_decoration_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &DecorationVersionRule,
        &DecorationGroupsRule,
        &BasicDecorationRule,
        &DecorationTargetCategoriesRule,
    ]
}
