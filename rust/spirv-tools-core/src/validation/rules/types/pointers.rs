//! Pointer type validation rules.
//!
//! This module validates SPIR-V pointer type requirements:
//! - OpTypePointer storage class and pointee type requirements
//! - OpTypeForwardPointer requirements
//! - OpTypeUntypedPointerKHR Vulkan requirements

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op, StorageClass};

use crate::target_env::TargetEnv;
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::{ResultId, TypeId};

use super::helpers::is_type_opcode;

// ============================================================================
// OpTypePointer Validation Rule
// ============================================================================

/// Validates OpTypePointer requirements.
///
/// Checks:
/// - Type operand must be a type instruction
/// - Storage class must be valid for the target environment
pub struct TypePointerRule;

impl ValidationRule for TypePointerRule {
    fn name(&self) -> &'static str {
        "type-pointer"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypePointer {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get storage class (operand 0)
            let storage_class = match inst.operands.first() {
                Some(Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            // Get pointee type (operand 1)
            let pointee_type_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate pointee type is a type instruction
            if let Ok(pointee_result_id) = ResultId::try_from(pointee_type_raw) {
                if let Some(pointee_opcode) = ctx.opcodes.get(&pointee_result_id) {
                    if !is_type_opcode(*pointee_opcode) {
                        let pointee_type = TypeId::try_from(pointee_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypePointerTypeNotType {
                            type_id,
                            pointee_type,
                        }.into());
                    }
                }
            }

            // Validate storage class for target environment
            if !is_valid_storage_class_for_env(storage_class, ctx.env) {
                return Err(ValidationError::TypePointerInvalidStorageClass {
                    type_id,
                    storage_class,
                }.into());
            }
        }

        Ok(())
    }
}

/// Checks if a storage class is valid for the target environment.
fn is_valid_storage_class_for_env(storage_class: StorageClass, env: TargetEnv) -> bool {
    // Most storage classes are universally valid
    // Only certain storage classes are restricted to specific environments
    match storage_class {
        // These are not allowed in Vulkan (Shader environments)
        StorageClass::Generic | StorageClass::AtomicCounter => {
            !matches!(
                env,
                TargetEnv::Vulkan1_0
                    | TargetEnv::Vulkan1_1
                    | TargetEnv::Vulkan1_1Spirv1_4
                    | TargetEnv::Vulkan1_2
                    | TargetEnv::Vulkan1_3
                    | TargetEnv::Vulkan1_4
            )
        }
        // All other storage classes are generally valid
        _ => true,
    }
}

// ============================================================================
// OpTypeForwardPointer Validation Rule
// ============================================================================

/// Validates OpTypeForwardPointer requirements.
///
/// Checks:
/// - Pointer type ID must refer to an OpTypePointer
/// - Storage class must match the pointer definition
/// - Forward pointer must point to a struct
/// - (Vulkan) Storage class must be PhysicalStorageBuffer
pub struct TypeForwardPointerRule;

impl ValidationRule for TypeForwardPointerRule {
    fn name(&self) -> &'static str {
        "type-forward-pointer"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let is_vulkan = matches!(
            ctx.env,
            TargetEnv::Vulkan1_0
                | TargetEnv::Vulkan1_1
                | TargetEnv::Vulkan1_1Spirv1_4
                | TargetEnv::Vulkan1_2
                | TargetEnv::Vulkan1_3
                | TargetEnv::Vulkan1_4
        );

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeForwardPointer {
                continue;
            }

            // Get pointer type ID (operand 0)
            let pointer_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            let target_type = TypeId::try_from(pointer_type_raw)
                .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());

            // Get storage class (operand 1)
            let forward_storage_class = match inst.operands.get(1) {
                Some(Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            // Get the pointer type instruction
            let pointer_result_id = match ResultId::try_from(pointer_type_raw) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let pointer_inst = match ctx.definitions.get(&pointer_result_id) {
                Some(inst) => inst,
                None => continue,
            };

            // Validate pointer type is OpTypePointer
            if pointer_inst.class.opcode != Op::TypePointer {
                return Err(ValidationError::ForwardPointerNotPointerType { target_type }.into());
            }

            // Get storage class from pointer definition (operand 0 of OpTypePointer)
            let pointer_storage_class = match pointer_inst.operands.first() {
                Some(Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            // Validate storage class matches
            if forward_storage_class != pointer_storage_class {
                return Err(ValidationError::ForwardPointerStorageClassMismatch {
                    target_type,
                    forward_storage_class,
                    pointer_storage_class,
                }.into());
            }

            // Get pointee type from pointer definition (operand 1 of OpTypePointer)
            let pointee_type_raw = match pointer_inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate pointee type is a struct
            if let Ok(pointee_result_id) = ResultId::try_from(pointee_type_raw) {
                if let Some(pointee_opcode) = ctx.opcodes.get(&pointee_result_id) {
                    if *pointee_opcode != Op::TypeStruct {
                        return Err(ValidationError::ForwardPointerNotPointingToStruct {
                            target_type,
                        }.into());
                    }
                }
            }

            // Vulkan: Storage class must be PhysicalStorageBuffer
            if is_vulkan && forward_storage_class != StorageClass::PhysicalStorageBuffer {
                return Err(ValidationError::ForwardPointerRequiresPhysicalStorageBuffer {
                    target_type,
                    storage_class: forward_storage_class,
                }.into());
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeUntypedPointerKHR Validation Rule
// ============================================================================

/// Validates OpTypeUntypedPointerKHR requirements.
///
/// Checks (Vulkan only):
/// - Workgroup storage class requires WorkgroupMemoryExplicitLayoutKHR capability
/// - Only certain storage classes are allowed (StorageBuffer, PhysicalStorageBuffer,
///   Uniform, PushConstant, Workgroup)
pub struct TypeUntypedPointerKHRRule;

impl ValidationRule for TypeUntypedPointerKHRRule {
    fn name(&self) -> &'static str {
        "type-untyped-pointer-khr"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let is_vulkan = matches!(
            ctx.env,
            TargetEnv::Vulkan1_0
                | TargetEnv::Vulkan1_1
                | TargetEnv::Vulkan1_1Spirv1_4
                | TargetEnv::Vulkan1_2
                | TargetEnv::Vulkan1_3
                | TargetEnv::Vulkan1_4
        );

        if !is_vulkan {
            return Ok(());
        }

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeUntypedPointerKHR {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get storage class (operand 0)
            let storage_class = match inst.operands.first() {
                Some(Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            match storage_class {
                StorageClass::Workgroup => {
                    if !ctx.has_capability(Capability::WorkgroupMemoryExplicitLayoutKHR) {
                        return Err(
                            ValidationError::TypeUntypedPointerWorkgroupRequiresCapability {
                                type_id,
                            }.into(),
                        );
                    }
                }
                StorageClass::StorageBuffer
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::Uniform
                | StorageClass::PushConstant => {
                    // These are valid in Vulkan
                }
                _ => {
                    return Err(ValidationError::TypeUntypedPointerInvalidStorageClass {
                        type_id,
                        storage_class,
                    }.into());
                }
            }
        }

        Ok(())
    }
}

/// Returns all pointer type validation rules.
pub fn all_pointer_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &TypePointerRule,
        &TypeForwardPointerRule,
        &TypeUntypedPointerKHRRule,
    ]
}
