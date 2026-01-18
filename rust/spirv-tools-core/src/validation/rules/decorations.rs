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
// Decoration Compatibility Rule
// ============================================================================

/// Validates that mutually exclusive decorations are not both applied.
pub struct DecorationCompatibilityRule;

impl ValidationRule for DecorationCompatibilityRule {
    fn name(&self) -> &'static str {
        "decoration-compatibility"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        // Mutually exclusive decorations per ID
        const MUTUALLY_EXCLUSIVE_PER_ID: &[(Decoration, Decoration)] = &[
            (Decoration::Block, Decoration::BufferBlock),
            (Decoration::Restrict, Decoration::Aliased),
            (Decoration::RestrictPointer, Decoration::AliasedPointer),
        ];

        // Mutually exclusive decorations per member
        const MUTUALLY_EXCLUSIVE_PER_MEMBER: &[(Decoration, Decoration)] = &[
            (Decoration::RowMajor, Decoration::ColMajor),
        ];

        // Track seen decorations per ID
        let mut seen_per_id: HashSet<(Decoration, u32)> = HashSet::new();
        // Track seen decorations per (ID, member)
        let mut seen_per_member: HashSet<(Decoration, u32, u32)> = HashSet::new();

        for inst in &module.annotations {
            match inst.class.opcode {
                Op::Decorate => {
                    let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                        continue;
                    };
                    let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                        continue;
                    };
                    let key = (*decoration, *target);

                    // Check for duplicate at-most-once decorations
                    if at_most_once_per_id(*decoration) && seen_per_id.contains(&key) {
                        return Err(ValidationError::DuplicateDecorationOnId {
                            decoration: *decoration,
                            target: *target,
                        });
                    }
                    seen_per_id.insert(key);

                    // Check for mutually exclusive decorations
                    for &(dec_a, dec_b) in MUTUALLY_EXCLUSIVE_PER_ID {
                        let excl_dec = if dec_a == *decoration {
                            Some(dec_b)
                        } else if dec_b == *decoration {
                            Some(dec_a)
                        } else {
                            None
                        };

                        if let Some(excl_dec) = excl_dec {
                            if seen_per_id.contains(&(excl_dec, *target)) {
                                return Err(ValidationError::MutuallyExclusiveDecorations {
                                    decoration1: *decoration,
                                    decoration2: excl_dec,
                                    target: *target,
                                });
                            }
                        }
                    }
                }
                Op::MemberDecorate => {
                    let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                        continue;
                    };
                    let Some(rspirv::dr::Operand::LiteralBit32(member)) = inst.operands.get(1) else {
                        continue;
                    };
                    let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(2) else {
                        continue;
                    };
                    let key = (*decoration, *target, *member);

                    // Check for duplicate at-most-once member decorations
                    if at_most_once_per_member(*decoration) && seen_per_member.contains(&key) {
                        return Err(ValidationError::DuplicateMemberDecoration {
                            decoration: *decoration,
                            target: *target,
                            member: *member,
                        });
                    }
                    seen_per_member.insert(key);

                    // Check for mutually exclusive member decorations
                    for &(dec_a, dec_b) in MUTUALLY_EXCLUSIVE_PER_MEMBER {
                        let excl_dec = if dec_a == *decoration {
                            Some(dec_b)
                        } else if dec_b == *decoration {
                            Some(dec_a)
                        } else {
                            None
                        };

                        if let Some(excl_dec) = excl_dec {
                            if seen_per_member.contains(&(excl_dec, *target, *member)) {
                                return Err(ValidationError::MutuallyExclusiveMemberDecorations {
                                    decoration1: *decoration,
                                    decoration2: excl_dec,
                                    target: *target,
                                    member: *member,
                                });
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

/// Returns true if the decoration can only be applied once per ID.
fn at_most_once_per_id(decoration: Decoration) -> bool {
    matches!(
        decoration,
        Decoration::Block
            | Decoration::BufferBlock
            | Decoration::RowMajor
            | Decoration::ColMajor
            | Decoration::ArrayStride
            | Decoration::MatrixStride
            | Decoration::BuiltIn
            | Decoration::NoPerspective
            | Decoration::Flat
            | Decoration::Patch
            | Decoration::Centroid
            | Decoration::Sample
            | Decoration::Invariant
            | Decoration::Restrict
            | Decoration::Aliased
            | Decoration::Volatile
            | Decoration::Coherent
            | Decoration::NonWritable
            | Decoration::NonReadable
            | Decoration::Uniform
            | Decoration::Location
            | Decoration::Component
            | Decoration::Index
            | Decoration::Binding
            | Decoration::DescriptorSet
            | Decoration::Offset
            | Decoration::XfbBuffer
            | Decoration::XfbStride
            | Decoration::NoContraction
            | Decoration::InputAttachmentIndex
            | Decoration::Alignment
    )
}

/// Returns true if the decoration can only be applied once per struct member.
fn at_most_once_per_member(decoration: Decoration) -> bool {
    matches!(
        decoration,
        Decoration::RowMajor
            | Decoration::ColMajor
            | Decoration::MatrixStride
            | Decoration::Offset
            | Decoration::Alignment
    )
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
// Linkage Attribute Rule
// ============================================================================

/// Validates linkage attributes on functions and variables.
///
/// - Function declarations (no basic blocks) must have Import linkage
/// - Function definitions (with basic blocks) must not have Import linkage
/// - Imported variables cannot have initializers
pub struct LinkageAttributeRule;

impl ValidationRule for LinkageAttributeRule {
    fn name(&self) -> &'static str {
        "linkage-attributes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        use crate::validation::types::Id;
        use rspirv::spirv::LinkageType;

        let module = ctx.module;

        // Only apply linkage validation if Linkage capability is declared
        // Without the Linkage capability, function declarations are not valid anyway
        // and will be caught by other validation rules
        let has_linkage_capability = ctx.declared_capabilities.contains(&Capability::Linkage);

        // Build a map of IDs with Import linkage
        let mut import_linkage_ids: HashSet<u32> = HashSet::new();
        let mut has_any_linkage_decoration = false;
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(Decoration::LinkageAttributes)) =
                inst.operands.get(1)
            else {
                continue;
            };
            has_any_linkage_decoration = true;
            // LinkageAttributes has: name (string), linkage type
            // The linkage type is typically the last operand
            for operand in inst.operands.iter().rev() {
                if let rspirv::dr::Operand::LinkageType(linkage) = operand {
                    if *linkage == LinkageType::Import {
                        import_linkage_ids.insert(*target);
                    }
                    break;
                }
            }
        }

        // Only enforce linkage rules if we have linkage capability or linkage decorations
        if !has_linkage_capability && !has_any_linkage_decoration {
            return Ok(());
        }

        // Check function declarations/definitions
        for func in &module.functions {
            let Some(func_id) = func.def.as_ref().and_then(|d| d.result_id) else {
                continue;
            };

            let has_import_linkage = import_linkage_ids.contains(&func_id);

            // A true declaration has no blocks AND no parameters in blocks
            // (a malformed function that's missing OpLabel is not a declaration)
            let is_declaration = func.blocks.is_empty() && func.parameters.is_empty();

            if is_declaration && !has_import_linkage && has_linkage_capability {
                // Function declarations must have Import linkage
                // However, entry points are exempt from this rule
                let is_entry_point = module.entry_points.iter().any(|ep| {
                    ep.operands.get(1).map(|op| {
                        if let rspirv::dr::Operand::IdRef(id) = op {
                            *id == func_id
                        } else {
                            false
                        }
                    }).unwrap_or(false)
                });

                if !is_entry_point {
                    return Err(ValidationError::FunctionDeclarationMissingImportLinkage {
                        function: Id::try_from(func_id).unwrap_or_else(|_| Id::try_from(1u32).unwrap()),
                    });
                }
            }

            // A function with blocks has a definition - check it doesn't have import linkage
            if !func.blocks.is_empty() && has_import_linkage {
                return Err(ValidationError::FunctionDefinitionHasImportLinkage {
                    function: Id::try_from(func_id).unwrap_or_else(|_| Id::try_from(1u32).unwrap()),
                });
            }
        }

        // Check imported variables for initializers
        for inst in &module.types_global_values {
            if inst.class.opcode != Op::Variable {
                continue;
            }
            let Some(var_id) = inst.result_id else {
                continue;
            };
            // OpVariable: result_type, result_id, storage_class, [initializer]
            // If there's an initializer (5th word / 4th operand index)
            let has_initializer = inst.operands.len() > 1; // storage_class + initializer

            if has_initializer && import_linkage_ids.contains(&var_id) {
                return Err(ValidationError::ImportedVariableHasInitializer {
                    variable: Id::try_from(var_id).unwrap_or_else(|_| Id::try_from(1u32).unwrap()),
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Vulkan Memory Model Deprecation Rule
// ============================================================================

/// Validates that deprecated decorations are not used with the Vulkan memory model.
pub struct VulkanMemoryModelDecorationRule;

impl ValidationRule for VulkanMemoryModelDecorationRule {
    fn name(&self) -> &'static str {
        "vulkan-memory-model-decorations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        use rspirv::spirv::MemoryModel;

        // Only check if using Vulkan memory model
        let uses_vulkan_memory_model = ctx.module.memory_model.as_ref().map(|mm| {
            mm.operands.get(1).map(|op| {
                matches!(op, rspirv::dr::Operand::MemoryModel(MemoryModel::Vulkan))
            }).unwrap_or(false)
        }).unwrap_or(false);

        if !uses_vulkan_memory_model {
            return Ok(());
        }

        // Check for deprecated decorations
        for inst in &ctx.module.annotations {
            if inst.class.opcode != Op::Decorate && inst.class.opcode != Op::MemberDecorate {
                continue;
            }
            let decoration_idx = if inst.class.opcode == Op::Decorate { 1 } else { 2 };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(decoration_idx) else {
                continue;
            };

            // Coherent and Volatile are deprecated with Vulkan memory model
            if matches!(decoration, Decoration::Coherent | Decoration::Volatile) {
                return Err(ValidationError::VulkanMemoryModelDeprecatesDecoration {
                    decoration: *decoration,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Integer Wrap Decoration Rule
// ============================================================================

/// Validates NoSignedWrap and NoUnsignedWrap decorations are only applied to integer operations.
pub struct IntegerWrapDecorationRule;

impl ValidationRule for IntegerWrapDecorationRule {
    fn name(&self) -> &'static str {
        "integer-wrap-decorations"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };

            if !matches!(decoration, Decoration::NoSignedWrap | Decoration::NoUnsignedWrap) {
                continue;
            }

            // Check that the target is an integer arithmetic operation
            let target_id = match ResultId::try_from(*target) {
                Ok(id) => id,
                Err(_) => continue,
            };
            let Some(opcode) = ctx.opcodes.get(&target_id) else {
                continue;
            };

            // Valid opcodes for NoSignedWrap/NoUnsignedWrap
            let is_valid_op = matches!(
                opcode,
                Op::IAdd
                    | Op::ISub
                    | Op::IMul
                    | Op::ShiftLeftLogical
                    | Op::SNegate
                    // Extended integer arithmetic
                    | Op::SMulExtended
                    | Op::UMulExtended
            );

            if !is_valid_op {
                return Err(ValidationError::IntegerWrapDecorationInvalidOp { opcode: *opcode });
            }
        }

        Ok(())
    }
}

// ============================================================================
// RelaxedPrecision Decoration Rule
// ============================================================================

/// Validates RelaxedPrecision decoration is only used with Shader capability.
pub struct RelaxedPrecisionDecorationRule;

impl ValidationRule for RelaxedPrecisionDecorationRule {
    fn name(&self) -> &'static str {
        "relaxed-precision-decoration"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let has_shader = ctx.declared_capabilities.contains(&Capability::Shader);

        for inst in &ctx.module.annotations {
            let is_relaxed_precision = match inst.class.opcode {
                Op::Decorate => inst
                    .operands
                    .get(1)
                    .map(|op| matches!(op, rspirv::dr::Operand::Decoration(Decoration::RelaxedPrecision)))
                    .unwrap_or(false),
                Op::MemberDecorate => inst
                    .operands
                    .get(2)
                    .map(|op| matches!(op, rspirv::dr::Operand::Decoration(Decoration::RelaxedPrecision)))
                    .unwrap_or(false),
                _ => false,
            };

            if is_relaxed_precision && !has_shader {
                return Err(ValidationError::RelaxedPrecisionRequiresShader);
            }
        }

        Ok(())
    }
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
        &DecorationCompatibilityRule,
        &LinkageAttributeRule,
        &VulkanMemoryModelDecorationRule,
        &IntegerWrapDecorationRule,
        &RelaxedPrecisionDecorationRule,
    ]
}
