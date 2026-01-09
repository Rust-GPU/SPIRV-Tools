//! Interpolation decoration validation rules.
//!
//! This module validates SPIR-V interpolation decoration requirements including:
//!
//! - Storage class restrictions for interpolation decorations
//! - Mutual exclusivity of interpolation decorations
//! - Entry point compatibility requirements
//! - Fragment input Flat decoration requirements

use std::collections::HashMap;

use rspirv::dr::Instruction;
use rspirv::spirv::{Capability, Decoration, ExecutionModel, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::{build_decoration_lookup, is_vulkan_env};
use crate::validation::types::ResultId;

// ============================================================================
// Interpolation Storage Class Rule
// ============================================================================

/// Validates that interpolation decorations are used with valid storage classes.
pub struct InterpolationStorageClassRule;

impl ValidationRule for InterpolationStorageClassRule {
    fn name(&self) -> &'static str {
        "interpolation-storage-classes"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let mut operands = inst.operands.iter();
            let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = operands.next() else {
                continue;
            };
            let decoration = *decoration;
            let is_interp_base = matches!(
                decoration,
                Decoration::NoPerspective
                    | Decoration::Flat
                    | Decoration::Patch
                    | Decoration::Centroid
                    | Decoration::Sample
            );
            if !is_interp_base {
                continue;
            }
            let Ok(id) = ResultId::try_from(*target) else {
                continue;
            };
            let Some(def_inst) = ctx.definitions.get(&id) else {
                continue;
            };
            if def_inst.class.opcode != Op::Variable
                && def_inst.class.opcode != Op::UntypedVariableKHR
            {
                continue;
            }
            let storage_class = def_inst
                .operands
                .first()
                .and_then(|op| match op {
                    rspirv::dr::Operand::StorageClass(sc) => Some(*sc),
                    _ => None,
                })
                .unwrap_or(StorageClass::Function);
            if storage_class != StorageClass::Input && storage_class != StorageClass::Output {
                return Err(
                    ValidationError::InterpolationDecorationInvalidStorageClass {
                        decoration,
                        storage_class,
                    },
                );
            }

            if decoration == Decoration::Sample
                && !ctx
                    .declared_capabilities
                    .contains(&Capability::SampleRateShading)
            {
                return Err(ValidationError::DecorationRequiresCapability {
                    decoration,
                    capability: Capability::SampleRateShading,
                });
            }

            if decoration != Decoration::Patch
                && !ctx.entry_models.contains(&ExecutionModel::Fragment)
            {
                return Err(ValidationError::InterpolationDecorationRequiresFragment { decoration });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Interpolation Exclusivity Rule
// ============================================================================

/// Validates that conflicting interpolation decorations are not applied together.
pub struct InterpolationExclusivityRule;

impl ValidationRule for InterpolationExclusivityRule {
    fn name(&self) -> &'static str {
        "interpolation-exclusivity"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;

        #[derive(Default)]
        struct InterpDecorations {
            base: Option<Decoration>,
            centroid_sample_patch: Option<Decoration>,
        }

        let mut seen: HashMap<ResultId, InterpDecorations> = HashMap::new();

        for inst in &module.annotations {
            if inst.class.opcode != Op::Decorate {
                continue;
            }
            let mut operands = inst.operands.iter();
            let Some(rspirv::dr::Operand::IdRef(target)) = operands.next() else {
                continue;
            };
            let Some(rspirv::dr::Operand::Decoration(decoration)) = operands.next() else {
                continue;
            };
            let decoration = *decoration;
            if !matches!(
                decoration,
                Decoration::Flat
                    | Decoration::NoPerspective
                    | Decoration::Centroid
                    | Decoration::Sample
                    | Decoration::Patch
            ) {
                continue;
            }
            let Ok(id) = ResultId::try_from(*target) else {
                continue;
            };
            let Some(def_inst) = ctx.definitions.get(&id) else {
                continue;
            };
            if def_inst.class.opcode != Op::Variable
                && def_inst.class.opcode != Op::UntypedVariableKHR
            {
                continue;
            }

            let entry = seen.entry(id).or_default();
            if matches!(decoration, Decoration::Flat | Decoration::NoPerspective) {
                if let Some(existing) = entry.base {
                    if existing != decoration {
                        return Err(ValidationError::InterpolationDecorationConflict {
                            decoration,
                            existing,
                        });
                    }
                } else {
                    entry.base = Some(decoration);
                }
            }

            if matches!(
                decoration,
                Decoration::Centroid | Decoration::Sample | Decoration::Patch
            ) {
                if let Some(existing) = entry.base {
                    if existing == Decoration::Flat {
                        return Err(ValidationError::InterpolationDecorationConflict {
                            decoration,
                            existing,
                        });
                    }
                }
                if let Some(existing) = entry.centroid_sample_patch {
                    if existing != decoration {
                        return Err(ValidationError::InterpolationDecorationConflict {
                            decoration,
                            existing,
                        });
                    }
                } else {
                    entry.centroid_sample_patch = Some(decoration);
                }
            }

            if matches!(decoration, Decoration::Flat) {
                if let Some(existing) = entry.centroid_sample_patch {
                    return Err(ValidationError::InterpolationDecorationConflict {
                        decoration,
                        existing,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Interpolation Entry Point Compatibility Rule
// ============================================================================

/// Validates interpolation decorations against entry point requirements.
pub struct InterpolationEntryPointRule;

impl ValidationRule for InterpolationEntryPointRule {
    fn name(&self) -> &'static str {
        "interpolation-entry-point"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_vulkan_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let module = ctx.module;
        let decoration_lookup = build_decoration_lookup(module);

        for entry in &module.entry_points {
            let Some(rspirv::dr::Operand::ExecutionModel(model)) = entry.operands.first() else {
                continue;
            };
            let model = *model;
            let interfaces = entry.operands.iter().skip(2).filter_map(|op| match op {
                rspirv::dr::Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            });
            for var_id in interfaces {
                let Some(var_inst) = ctx.definitions.get(&var_id) else {
                    continue;
                };
                if var_inst.class.opcode != Op::Variable
                    && var_inst.class.opcode != Op::UntypedVariableKHR
                {
                    continue;
                }
                let storage_class = match var_inst.operands.first() {
                    Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                    _ => continue,
                };
                let decos = decoration_lookup.get(&var_id).cloned().unwrap_or_default();
                let has_interp = decos.contains(&Decoration::NoPerspective)
                    || decos.contains(&Decoration::Flat)
                    || decos.contains(&Decoration::Sample)
                    || decos.contains(&Decoration::Centroid);
                if has_interp {
                    match storage_class {
                        StorageClass::Input if model == ExecutionModel::Vertex => {
                            return Err(
                                ValidationError::InterpolationDecorationInvalidForEntryPoint {
                                    decoration: *decos
                                        .iter()
                                        .find(|d| {
                                            matches!(
                                                d,
                                                Decoration::NoPerspective
                                                    | Decoration::Flat
                                                    | Decoration::Sample
                                                    | Decoration::Centroid
                                            )
                                        })
                                        .unwrap(),
                                    storage_class,
                                    execution_model: model,
                                },
                            );
                        }
                        StorageClass::Output if model == ExecutionModel::Fragment => {
                            return Err(
                                ValidationError::InterpolationDecorationInvalidForEntryPoint {
                                    decoration: *decos
                                        .iter()
                                        .find(|d| {
                                            matches!(
                                                d,
                                                Decoration::NoPerspective
                                                    | Decoration::Flat
                                                    | Decoration::Sample
                                                    | Decoration::Centroid
                                            )
                                        })
                                        .unwrap(),
                                    storage_class,
                                    execution_model: model,
                                },
                            );
                        }
                        _ => {}
                    }
                }

                if model == ExecutionModel::Fragment && storage_class == StorageClass::Input {
                    let has_flat = decos.contains(&Decoration::Flat);
                    if !has_flat && fragment_requires_flat(var_inst, ctx.definitions) {
                        return Err(ValidationError::FragmentInputRequiresFlat);
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

/// Resolves the pointee type for a variable (through pointer indirection).
fn resolve_builtin_pointee_type<'a>(
    definitions: &'a HashMap<ResultId, Instruction>,
    var_id: ResultId,
) -> Option<&'a Instruction> {
    let var_inst = definitions.get(&var_id)?;
    let ptr_type_id = var_inst.result_type?;
    let ptr_type = ResultId::try_from(ptr_type_id)
        .ok()
        .and_then(|id| definitions.get(&id))?;
    if ptr_type.class.opcode != Op::TypePointer {
        return None;
    }
    let pointee_id = match ptr_type.operands.get(1) {
        Some(rspirv::dr::Operand::IdRef(id)) => *id,
        _ => return None,
    };
    ResultId::try_from(pointee_id)
        .ok()
        .and_then(|id| definitions.get(&id))
}

/// Determines if a fragment input variable requires the Flat decoration.
fn fragment_requires_flat(
    var_inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    let Some(var_id) = var_inst
        .result_id
        .and_then(|id| ResultId::try_from(id).ok())
    else {
        return false;
    };
    let Some(pointee) = resolve_builtin_pointee_type(definitions, var_id) else {
        return false;
    };
    is_int_scalar_or_vector(pointee, definitions)
        || is_float_scalar_of_width(pointee, definitions, 64)
}

/// Checks if a type is an integer scalar or vector.
fn is_int_scalar_or_vector(ty: &Instruction, definitions: &HashMap<ResultId, Instruction>) -> bool {
    match ty.class.opcode {
        Op::TypeInt => true,
        Op::TypeVector => {
            let Some(rspirv::dr::Operand::IdRef(elem)) = ty.operands.first() else {
                return false;
            };
            ResultId::try_from(*elem)
                .ok()
                .and_then(|id| definitions.get(&id))
                .is_some_and(|inst| inst.class.opcode == Op::TypeInt)
        }
        _ => false,
    }
}

/// Gets the bit width of a scalar type.
fn type_bit_width(ty: &Instruction) -> Option<u32> {
    ty.operands.first().and_then(|op| match op {
        rspirv::dr::Operand::LiteralBit32(w) => Some(*w),
        _ => None,
    })
}

/// Checks if a type is a float scalar (or vector of floats) with specified width.
fn is_float_scalar_of_width(
    ty: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    width: u32,
) -> bool {
    match ty.class.opcode {
        Op::TypeFloat => type_bit_width(ty) == Some(width),
        Op::TypeVector => {
            let Some(rspirv::dr::Operand::IdRef(elem)) = ty.operands.first() else {
                return false;
            };
            ResultId::try_from(*elem)
                .ok()
                .and_then(|id| definitions.get(&id))
                .is_some_and(|inst| {
                    inst.class.opcode == Op::TypeFloat && type_bit_width(inst) == Some(width)
                })
        }
        _ => false,
    }
}

// ============================================================================
// All interpolation rules
// ============================================================================

/// Returns all interpolation validation rules.
pub fn all_interpolation_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &InterpolationStorageClassRule,
        &InterpolationExclusivityRule,
        &InterpolationEntryPointRule,
    ]
}
