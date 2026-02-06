//! Storage class validation rules.
//!
//! This module validates SPIR-V storage class requirements including:
//!
//! - Block/BufferBlock decoration storage class compatibility
//! - Descriptor binding storage class requirements
//! - Location decoration storage class requirements
//! - Struct block decoration requirements

use std::collections::{HashMap, HashSet};

use rspirv::dr::Module;
use rspirv::spirv::{Capability, Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{build_decoration_lookup, is_vulkan_env};
use crate::validation::types::{Id, ResultId};
use crate::validation::ValidationResult;
use crate::version::SpirvVersion;

// ============================================================================
// Block Storage Class Rule
// ============================================================================

/// Validates that Block/BufferBlock decorations are used with valid storage classes.
pub struct BlockStorageClassRule;

impl ValidationRule for BlockStorageClassRule {
    fn name(&self) -> &'static str {
        "block-storage-classes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;
        let target_version = ctx.target_version;

        let mut blocks: HashMap<ResultId, (Decoration, HashSet<StorageClass>)> = HashMap::new();
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(decoration)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                if *decoration == Decoration::Block || *decoration == Decoration::BufferBlock {
                    if let Ok(id) = ResultId::try_from(*target) {
                        blocks.entry(id).or_insert((*decoration, HashSet::new()));
                    }
                }
            }
        }

        if blocks.is_empty() {
            return Ok(());
        }

        for var in &module.types_global_values {
            if var.class.opcode != Op::Variable {
                continue;
            }
            let Some(rspirv::dr::Operand::StorageClass(storage_class)) = var.operands.first()
            else {
                continue;
            };
            let Some(result_type) = var.result_type else {
                continue;
            };
            let Ok(ptr_type) = ResultId::try_from(result_type) else {
                continue;
            };
            let Some(ptr_inst) = module
                .types_global_values
                .iter()
                .find(|inst| inst.result_id == Some(u32::from(ptr_type)))
            else {
                continue;
            };
            if ptr_inst.class.opcode != Op::TypePointer {
                continue;
            }
            let pointee = match ptr_inst.operands.get(1) {
                Some(rspirv::dr::Operand::IdRef(raw)) => match ResultId::try_from(*raw) {
                    Ok(id) => id,
                    Err(_) => continue,
                },
                _ => continue,
            };
            if let Some((decoration, classes)) = blocks.get_mut(&pointee) {
                classes.insert(*storage_class);
                // Early version gate: BufferBlock was replaced after 1.3.
                if *decoration == Decoration::BufferBlock
                    && target_version > SpirvVersion::new(1, 3)
                {
                    return Err(ValidationError::DecorationRequiresSpirvVersion {
                        decoration: *decoration,
                        required_version: SpirvVersion::new(1, 3),
                        target_version,
                    }
                    .into());
                }
            }
        }

        let workgroup_blocks_allowed =
            ctx.has_capability(Capability::WorkgroupMemoryExplicitLayoutKHR);

        for (_block_id, (decoration, storage_classes)) in blocks {
            if storage_classes.is_empty() {
                continue;
            }
            for storage_class in storage_classes {
                let allowed = match decoration {
                    Decoration::Block => {
                        matches!(
                            storage_class,
                            StorageClass::Uniform
                                | StorageClass::StorageBuffer
                                | StorageClass::PhysicalStorageBuffer
                                | StorageClass::PushConstant
                                | StorageClass::Input
                                | StorageClass::Output
                        ) || (storage_class == StorageClass::Workgroup && workgroup_blocks_allowed)
                    }
                    Decoration::BufferBlock => matches!(
                        storage_class,
                        StorageClass::Uniform
                            | StorageClass::StorageBuffer
                            | StorageClass::PhysicalStorageBuffer
                    ),
                    _ => true,
                };
                if !allowed {
                    return Err(ValidationError::InvalidBlockDecorationStorageClass {
                        decoration,
                        storage_class,
                    }
                    .into());
                }

                // PushConstant requires Block, never BufferBlock.
                if storage_class == StorageClass::PushConstant
                    && decoration == Decoration::BufferBlock
                {
                    return Err(ValidationError::InvalidBlockDecorationStorageClass {
                        decoration,
                        storage_class,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Descriptor Storage Class Rule
// ============================================================================

/// Validates that descriptor-decorated variables use valid storage classes.
pub struct DescriptorStorageClassRule;

impl ValidationRule for DescriptorStorageClassRule {
    fn name(&self) -> &'static str {
        "descriptor-storage-classes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;

        let mut decorated_vars: HashMap<ResultId, StorageClass> = HashMap::new();

        for inst in &module.types_global_values {
            if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
                continue;
            }
            let Some(rspirv::dr::Operand::StorageClass(sc)) = inst.operands.first() else {
                continue;
            };
            if let Some(result_id) = inst.result_id {
                if let Ok(id) = ResultId::try_from(result_id) {
                    decorated_vars.insert(id, *sc);
                }
            }
        }

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            if *decoration != Decoration::Binding && *decoration != Decoration::DescriptorSet {
                continue;
            }
            let Ok(var_id) = ResultId::try_from(*target) else {
                continue;
            };
            let Some(storage_class) = decorated_vars.get(&var_id) else {
                continue;
            };

            let allowed = matches!(
                storage_class,
                StorageClass::UniformConstant
                    | StorageClass::Uniform
                    | StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
            );
            if !allowed {
                return Err(ValidationError::InvalidDescriptorStorageClass {
                    storage_class: *storage_class,
                }
                .into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Descriptor Requirements Rule
// ============================================================================

/// Validates that interface variables have required descriptor decorations.
pub struct DescriptorRequirementsRule;

impl ValidationRule for DescriptorRequirementsRule {
    fn name(&self) -> &'static str {
        "descriptor-requirements"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;

        let interface_vars: HashSet<ResultId> = module
            .entry_points
            .iter()
            .flat_map(|ep| ep.operands.iter().skip(2))
            .filter_map(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })
            .collect();

        let decoration_lookup = build_decoration_lookup(module);

        for var in &module.types_global_values {
            if var.class.opcode != Op::Variable && var.class.opcode != Op::UntypedVariableKHR {
                continue;
            }
            let Some(raw_id) = var.result_id else {
                continue;
            };
            let Some(rid) = ResultId::try_from(raw_id).ok() else {
                continue;
            };
            if !interface_vars.contains(&rid) {
                continue;
            }
            let Some(storage_class) = var.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            }) else {
                continue;
            };
            if !matches!(
                storage_class,
                StorageClass::UniformConstant
                    | StorageClass::Uniform
                    | StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
            ) {
                continue;
            }
            let decos = decoration_lookup.get(&rid).cloned().unwrap_or_default();
            if decos.contains(&Decoration::BuiltIn) {
                continue;
            }
            let has_descriptor_set = decos.contains(&Decoration::DescriptorSet);
            let has_binding = decos.contains(&Decoration::Binding);
            if !has_descriptor_set {
                return Err(ValidationError::MissingDescriptorSetDecoration {
                    variable: Id::from(rid),
                }
                .into());
            }
            if !has_binding {
                return Err(ValidationError::MissingBindingDecoration {
                    variable: Id::from(rid),
                }
                .into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// Struct Block Requirements Rule
// ============================================================================

/// Validates struct block decoration requirements.
pub struct StructBlockRequirementsRule;

impl ValidationRule for StructBlockRequirementsRule {
    fn name(&self) -> &'static str {
        "struct-block-requirements"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;
        let target_version = ctx.target_version;

        for var in &module.types_global_values {
            if var.class.opcode != Op::Variable {
                continue;
            }
            let Some(storage_class) = var.operands.first().and_then(|op| match op {
                rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            }) else {
                continue;
            };
            if !matches!(
                storage_class,
                StorageClass::Uniform
                    | StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::PushConstant
            ) {
                continue;
            }
            let Some(ptr_type) = var.result_type else {
                continue;
            };
            let Ok(ptr_id) = ResultId::try_from(ptr_type) else {
                continue;
            };
            let Some(ptr_inst) = module
                .types_global_values
                .iter()
                .find(|inst| inst.result_id == Some(u32::from(ptr_id)))
            else {
                continue;
            };
            if ptr_inst.class.opcode != Op::TypePointer {
                continue;
            }
            let pointee = match ptr_inst.operands.get(1) {
                Some(rspirv::dr::Operand::IdRef(id)) => match ResultId::try_from(*id) {
                    Ok(id) => id,
                    Err(_) => continue,
                },
                _ => continue,
            };
            let Some(type_inst) = module
                .types_global_values
                .iter()
                .find(|inst| inst.result_id == Some(u32::from(pointee)))
            else {
                continue;
            };
            if type_inst.class.opcode != Op::TypeStruct {
                continue;
            }

            let has_block = has_block_decoration(module, pointee);
            if has_block {
                if storage_class == StorageClass::PushConstant {
                    let has_buffer_block = has_buffer_block_decoration(module, pointee);
                    if has_buffer_block {
                        return Err(ValidationError::InvalidBlockDecorationStorageClass {
                            decoration: Decoration::BufferBlock,
                            storage_class,
                        }
                        .into());
                    }
                }
            } else {
                // Uniform and StorageBuffer require Block/BufferBlock decoration
                if storage_class == StorageClass::Uniform
                    || storage_class == StorageClass::PushConstant
                {
                    return Err(ValidationError::MissingBlockDecoration { storage_class }.into());
                }
                // StorageBuffer requires Block decoration after SPIR-V 1.3
                if storage_class == StorageClass::StorageBuffer
                    && target_version > SpirvVersion::new(1, 3)
                {
                    return Err(ValidationError::MissingBlockDecoration { storage_class }.into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Location Storage Class Rule
// ============================================================================

/// Validates that Location and Component decorations are used with valid storage classes.
pub struct LocationStorageClassRule;

impl ValidationRule for LocationStorageClassRule {
    fn name(&self) -> &'static str {
        "location-storage-classes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module;

        let mut var_storage_classes: HashMap<ResultId, StorageClass> = HashMap::new();
        for inst in &module.types_global_values {
            if inst.class.opcode != Op::Variable {
                continue;
            }
            if let (Some(result_id), Some(rspirv::dr::Operand::StorageClass(sc))) =
                (inst.result_id, inst.operands.first())
            {
                if let Ok(id) = ResultId::try_from(result_id) {
                    var_storage_classes.insert(id, *sc);
                }
            }
        }

        // Collect which variables have Location decoration
        let mut has_location: HashSet<ResultId> = HashSet::new();
        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            if let (
                Some(rspirv::dr::Operand::IdRef(target)),
                Some(rspirv::dr::Operand::Decoration(Decoration::Location)),
            ) = (inst.operands.first(), inst.operands.get(1))
            {
                if let Ok(var_id) = ResultId::try_from(*target) {
                    has_location.insert(var_id);
                }
            }
        }

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let Some(rspirv::dr::Operand::IdRef(target)) = inst.operands.first() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = inst.operands.get(1) else {
                continue;
            };
            let Ok(var_id) = ResultId::try_from(*target) else {
                continue;
            };

            match decoration {
                Decoration::Location => {
                    let Some(storage_class) = var_storage_classes.get(&var_id) else {
                        continue;
                    };
                    let allowed =
                        matches!(storage_class, StorageClass::Input | StorageClass::Output);
                    if !allowed {
                        return Err(ValidationError::InvalidLocationStorageClass {
                            storage_class: *storage_class,
                        }
                        .into());
                    }
                }
                Decoration::Component => {
                    // Component decoration must have a valid value (0-3)
                    if let Some(rspirv::dr::Operand::LiteralBit32(component)) = inst.operands.get(2)
                    {
                        if *component > 3 {
                            return Err(ValidationError::ComponentOutOfRange {
                                component: *component,
                            }
                            .into());
                        }
                    }

                    // Component decoration requires Location decoration
                    if !has_location.contains(&var_id) {
                        return Err(ValidationError::ComponentMissingLocation.into());
                    }

                    // Component decoration also requires Input/Output storage class
                    let Some(storage_class) = var_storage_classes.get(&var_id) else {
                        continue;
                    };
                    let allowed =
                        matches!(storage_class, StorageClass::Input | StorageClass::Output);
                    if !allowed {
                        return Err(ValidationError::InvalidLocationStorageClass {
                            storage_class: *storage_class,
                        }
                        .into());
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn has_block_decoration(module: &Module, type_id: ResultId) -> bool {
    module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(target)),
                    Some(rspirv::dr::Operand::Decoration(dec))
                ) if *target == u32::from(type_id)
                    && (*dec == Decoration::Block || *dec == Decoration::BufferBlock)
            )
    })
}

fn has_buffer_block_decoration(module: &Module, type_id: ResultId) -> bool {
    module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && matches!(
                (inst.operands.first(), inst.operands.get(1)),
                (
                    Some(rspirv::dr::Operand::IdRef(target)),
                    Some(rspirv::dr::Operand::Decoration(Decoration::BufferBlock))
                ) if *target == u32::from(type_id)
            )
    })
}

// ============================================================================
// All storage class rules
// ============================================================================

/// Returns all storage class validation rules.
pub fn all_storage_class_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &BlockStorageClassRule,
        &DescriptorStorageClassRule,
        &DescriptorRequirementsRule,
        &StructBlockRequirementsRule,
        &LocationStorageClassRule,
    ]
}
