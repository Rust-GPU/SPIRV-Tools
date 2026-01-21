//! Atomic instruction validation rules.
//!
//! This module validates SPIR-V atomic instructions including:
//!
//! - OpAtomicLoad, OpAtomicStore, OpAtomicExchange
//! - OpAtomicCompareExchange, OpAtomicCompareExchangeWeak
//! - OpAtomicIIncrement, OpAtomicIDecrement
//! - OpAtomicIAdd, OpAtomicISub
//! - OpAtomicSMin, OpAtomicUMin, OpAtomicSMax, OpAtomicUMax
//! - OpAtomicAnd, OpAtomicOr, OpAtomicXor
//! - OpAtomicFlagTestAndSet, OpAtomicFlagClear
//! - OpAtomicFAddEXT, OpAtomicFMinEXT, OpAtomicFMaxEXT
//!
//! Atomic operations have strict requirements for:
//! - Result type (integer, float, or bool depending on operation)
//! - Pointer storage class
//! - Capability requirements (Int64Atomics, AtomicFloat*EXT)

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Op, StorageClass};

use std::collections::HashMap;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_vulkan_env;
use crate::validation::op_ext::OpExt;
use crate::validation::type_ext::{DefaultTypeResolver, TypeInstructionExt, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId};
use rspirv::dr::Instruction;

// Note: Basic atomic opcode classification is provided by OpExt::is_atomic().
// The following helpers are specialized for atomic validation:

/// Returns true if the atomic opcode returns a result.
fn has_return_type(op: Op) -> bool {
    !matches!(op, Op::AtomicStore | Op::AtomicFlagClear)
}

/// Returns true if the atomic opcode requires a float result type.
fn requires_float_result(op: Op) -> bool {
    matches!(op, Op::AtomicFAddEXT | Op::AtomicFMinEXT | Op::AtomicFMaxEXT)
}

/// Returns true if the atomic opcode requires an integer result type.
fn requires_int_result(op: Op) -> bool {
    matches!(
        op,
        Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
    )
}

/// Returns true if the atomic opcode can have int or float result.
fn allows_int_or_float_result(op: Op) -> bool {
    matches!(op, Op::AtomicLoad | Op::AtomicExchange)
}

/// Returns true if the atomic opcode requires a bool result type.
fn requires_bool_result(op: Op) -> bool {
    matches!(op, Op::AtomicFlagTestAndSet)
}

/// Returns true if the type is a float16 vector with 2 or 4 components.
///
/// This is used for checking if AtomicFloat16VectorNV capability is required
/// for float atomic operations with 16-bit float vector types.
fn is_float16_vec2_or_vec4(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    let Ok(result_id) = ResultId::try_from(type_id) else {
        return false;
    };
    let Some(type_inst) = definitions.get(&result_id) else {
        return false;
    };

    // Check if it's a vector type
    if !type_inst.is_vector_type() {
        return false;
    }

    // Check component count is 2 or 4
    let Some(count) = type_inst.vector_component_count() else {
        return false;
    };
    if count != 2 && count != 4 {
        return false;
    }

    // Check component type is 16-bit float
    let Some(component_type_id) = type_inst.vector_component_type_id() else {
        return false;
    };
    let Ok(component_result_id) = ResultId::try_from(component_type_id) else {
        return false;
    };
    let Some(component_inst) = definitions.get(&component_result_id) else {
        return false;
    };

    // Must be float type with 16-bit width
    component_inst.is_float_type() && component_inst.numeric_bit_width() == Some(16)
}

// ============================================================================
// Storage Class Validation Extension Trait
// ============================================================================

/// Extension trait for atomic storage class validation.
pub trait AtomicStorageClassExt {
    /// Returns true if this storage class is allowed for atomics universally.
    fn is_atomic_allowed_universal(self) -> bool;

    /// Returns true if this storage class is allowed for atomics in Vulkan.
    fn is_atomic_allowed_vulkan(self) -> bool;

    /// Returns true if this storage class is allowed for atomics in OpenCL.
    fn is_atomic_allowed_opencl(self) -> bool;
}

impl AtomicStorageClassExt for StorageClass {
    fn is_atomic_allowed_universal(self) -> bool {
        matches!(
            self,
            StorageClass::Uniform
                | StorageClass::StorageBuffer
                | StorageClass::Workgroup
                | StorageClass::CrossWorkgroup
                | StorageClass::Generic
                | StorageClass::AtomicCounter
                | StorageClass::Image
                | StorageClass::Function
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::TaskPayloadWorkgroupEXT
        )
    }

    fn is_atomic_allowed_vulkan(self) -> bool {
        matches!(
            self,
            StorageClass::Uniform
                | StorageClass::StorageBuffer
                | StorageClass::Workgroup
                | StorageClass::Image
                | StorageClass::PhysicalStorageBuffer
                | StorageClass::TaskPayloadWorkgroupEXT
        )
    }

    fn is_atomic_allowed_opencl(self) -> bool {
        matches!(
            self,
            StorageClass::Function
                | StorageClass::Workgroup
                | StorageClass::CrossWorkgroup
                | StorageClass::Generic
        )
    }
}

// ============================================================================
// Atomic Result Type Rule
// ============================================================================

/// Validates that atomic instructions have correct result types.
///
/// Ensures that:
/// - Float atomics (FAdd, FMin, FMax) return float scalars
/// - Integer atomics return integer scalars
/// - Load/Exchange can return int or float
/// - FlagTestAndSet returns bool
pub struct AtomicResultTypeRule;

impl ValidationRule for AtomicResultTypeRule {
    fn name(&self) -> &'static str {
        "atomic-result-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !inst.class.opcode.is_atomic() {
                        continue;
                    }

                    if !has_return_type(inst.class.opcode) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    let op = inst.class.opcode;

                    // Check result type requirements based on opcode
                    if requires_float_result(op) {
                        if !resolver.is_float_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "float scalar",
                                }.into());
                            }
                        }
                    } else if requires_int_result(op) {
                        if !resolver.is_int_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "integer scalar",
                                }.into());
                            }
                        }
                    } else if allows_int_or_float_result(op) {
                        let is_int = resolver.is_int_scalar(result_type_id, ctx.definitions);
                        let is_float = resolver.is_float_scalar(result_type_id, ctx.definitions);
                        if !is_int && !is_float {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "integer or float scalar",
                                }.into());
                            }
                        }
                    } else if requires_bool_result(op) {
                        if !resolver.is_bool_scalar(result_type_id, ctx.definitions) {
                            if let (Some(func), Some(block), Ok(result_type)) = (
                                function_id,
                                block_id,
                                TypeId::try_from(result_type_id),
                            ) {
                                return Err(ValidationError::AtomicResultTypeInvalid {
                                    function: func,
                                    block,
                                    opcode: op,
                                    result_type,
                                    expected: "bool scalar",
                                }.into());
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Storage Class Rule
// ============================================================================

/// Validates that atomic instructions use valid storage classes.
///
/// Different environments have different allowed storage classes:
/// - Universal: Uniform, StorageBuffer, Workgroup, CrossWorkgroup, Generic,
///   AtomicCounter, Image, Function, PhysicalStorageBuffer, TaskPayloadWorkgroupEXT
/// - Vulkan: Uniform, StorageBuffer, Workgroup, Image, PhysicalStorageBuffer,
///   TaskPayloadWorkgroupEXT
/// - Shader capability: Function storage class is forbidden
pub struct AtomicStorageClassRule;

impl ValidationRule for AtomicStorageClassRule {
    fn name(&self) -> &'static str {
        "atomic-storage-class"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let is_vulkan = is_vulkan_env(ctx.env);
        let has_shader = ctx.has_capability(Capability::Shader);

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !inst.class.opcode.is_atomic() {
                        continue;
                    }

                    // Get the pointer operand (first operand for store/clear, after result for others)
                    let pointer_operand_idx = if has_return_type(inst.class.opcode) {
                        0
                    } else {
                        0
                    };

                    let pointer_id = inst.operands.get(pointer_operand_idx).and_then(|op| {
                        match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        }
                    });

                    let Some(pointer_id) = pointer_id else {
                        continue;
                    };

                    // Get the type of the pointer
                    let pointer_type_id = ResultId::try_from(pointer_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type);

                    let Some(pointer_type_id) = pointer_type_id else {
                        continue;
                    };

                    // Look up the pointer type to get storage class
                    let pointer_type_inst = ResultId::try_from(pointer_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(pointer_type_inst) = pointer_type_inst else {
                        continue;
                    };

                    if pointer_type_inst.class.opcode != Op::TypePointer {
                        continue;
                    }

                    // Storage class is the first operand of OpTypePointer
                    let storage_class = pointer_type_inst.operands.first().and_then(|op| {
                        match op {
                            Operand::StorageClass(sc) => Some(*sc),
                            _ => None,
                        }
                    });

                    let Some(storage_class) = storage_class else {
                        continue;
                    };

                    // Check universal rules
                    if !storage_class.is_atomic_allowed_universal() {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "forbidden by universal validation rules",
                            }.into());
                        }
                    }

                    // Check Vulkan rules
                    if is_vulkan && !storage_class.is_atomic_allowed_vulkan() {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "forbidden in Vulkan environment",
                            }.into());
                        }
                    }

                    // Check OpenCL rules
                    if ctx.env.is_opencl() {
                        if !storage_class.is_atomic_allowed_opencl() {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::AtomicStorageClassForbidden {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    storage_class,
                                    reason: "storage class must be Function, Workgroup, CrossWorkgroup or Generic in the OpenCL environment",
                                }.into());
                            }
                        }

                        // OpenCL 1.2 forbids Generic storage class
                        if ctx.env.is_opencl_1_2() && storage_class == StorageClass::Generic {
                            if let (Some(func), Some(block)) = (function_id, block_id) {
                                return Err(ValidationError::AtomicStorageClassForbidden {
                                    function: func,
                                    block,
                                    opcode: inst.class.opcode,
                                    storage_class,
                                    reason: "storage class cannot be Generic in OpenCL 1.2 environment",
                                }.into());
                            }
                        }
                    }

                    // Shader capability forbids Function storage class (except Vulkan)
                    if has_shader && !is_vulkan && storage_class == StorageClass::Function {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorageClassForbidden {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                storage_class,
                                reason: "Function storage class forbidden when Shader capability is declared",
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Int64 Capability Rule
// ============================================================================

/// Validates that 64-bit integer atomics require Int64Atomics capability.
pub struct AtomicInt64CapabilityRule;

impl ValidationRule for AtomicInt64CapabilityRule {
    fn name(&self) -> &'static str {
        "atomic-int64-capability"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // If we already have Int64Atomics, everything is fine
        if ctx.has_capability(Capability::Int64Atomics) {
            return Ok(());
        }

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !inst.class.opcode.is_atomic() {
                        continue;
                    }

                    // Check if this is a 64-bit integer atomic
                    let data_type_id = if has_return_type(inst.class.opcode) {
                        inst.result_type
                    } else {
                        // For store operations, get the value type
                        inst.operands.get(3).and_then(|op| match op {
                            Operand::IdRef(id) => ResultId::try_from(*id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid))
                                .and_then(|inst| inst.result_type),
                            _ => None,
                        })
                    };

                    let Some(data_type_id) = data_type_id else {
                        continue;
                    };

                    // Check if it's a 64-bit integer
                    if resolver.is_int_scalar(data_type_id, ctx.definitions)
                        && resolver.get_bit_width(data_type_id, ctx.definitions) == Some(64)
                    {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicMissingCapability {
                                function: func,
                                block,
                                opcode: inst.class.opcode,
                                required_capability: Capability::Int64Atomics,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Flag Pointer Type Rule
// ============================================================================

/// Validates that atomic flag instructions point to 32-bit integers.
///
/// OpAtomicFlagTestAndSet and OpAtomicFlagClear require their pointer operand
/// to point to a 32-bit integer type.
pub struct AtomicFlagPointerTypeRule;

impl ValidationRule for AtomicFlagPointerTypeRule {
    fn name(&self) -> &'static str {
        "atomic-flag-pointer-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Only check flag operations
                    if !matches!(op, Op::AtomicFlagTestAndSet | Op::AtomicFlagClear) {
                        continue;
                    }

                    // Get pointer operand (first operand for both)
                    let pointer_id = inst.operands.first().and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });

                    let Some(pointer_id) = pointer_id else {
                        continue;
                    };

                    // Get the type of the pointer
                    let pointer_type_id = ResultId::try_from(pointer_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                        .and_then(|inst| inst.result_type);

                    let Some(pointer_type_id) = pointer_type_id else {
                        continue;
                    };

                    // Look up the pointer type to get pointee type
                    let pointer_type_inst = ResultId::try_from(pointer_type_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(pointer_type_inst) = pointer_type_inst else {
                        continue;
                    };

                    // Check for untyped pointers - they are not supported by atomic flag instructions
                    if pointer_type_inst.class.opcode == Op::TypeUntypedPointerKHR {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicFlagUntypedPointerNotSupported {
                                function: func,
                                block,
                                opcode: op,
                            }.into());
                        }
                        continue;
                    }

                    if pointer_type_inst.class.opcode != Op::TypePointer {
                        continue;
                    }

                    // Get the pointee type (second operand of OpTypePointer)
                    let pointee_type_id = pointer_type_inst.operands.get(1).and_then(|op| match op
                    {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });

                    let Some(pointee_type_id) = pointee_type_id else {
                        continue;
                    };

                    // Check if pointee is 32-bit integer
                    let is_int = resolver.is_int_scalar(pointee_type_id, ctx.definitions);
                    let bit_width = resolver.get_bit_width(pointee_type_id, ctx.definitions);

                    if !is_int || bit_width != Some(32) {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicFlagPointerTypeMismatch {
                                function: func,
                                block,
                                opcode: op,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Float Capability Rule
// ============================================================================

/// Validates that float atomic operations require the correct capabilities.
///
/// Float atomics require specific capabilities based on the bit width:
/// - OpAtomicFAddEXT: AtomicFloat16AddEXT (16-bit scalar), AtomicFloat32AddEXT (32-bit),
///   AtomicFloat64AddEXT (64-bit)
/// - OpAtomicFMinEXT/OpAtomicFMaxEXT: AtomicFloat16MinMaxEXT (16-bit scalar),
///   AtomicFloat32MinMaxEXT (32-bit), AtomicFloat64MinMaxEXT (64-bit)
/// - 16-bit float vec2/vec4 types require AtomicFloat16VectorNV for all float atomic ops
pub struct AtomicFloatCapabilityRule;

impl ValidationRule for AtomicFloatCapabilityRule {
    fn name(&self) -> &'static str {
        "atomic-float-capability"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Only applies when Shader capability is present
        if !ctx.has_capability(Capability::Shader) {
            return Ok(());
        }

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Only check float atomic operations
                    if !matches!(op, Op::AtomicFAddEXT | Op::AtomicFMinEXT | Op::AtomicFMaxEXT) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Get the bit width
                    let bit_width = resolver.get_bit_width(result_type_id, ctx.definitions);

                    let Some(bit_width) = bit_width else {
                        continue;
                    };

                    // Check if result type is a float16 vec2 or vec4
                    let is_f16_vector = is_float16_vec2_or_vec4(result_type_id, ctx.definitions);

                    // Determine required capability based on operation and bit width
                    let required_capability = match (op, bit_width) {
                        (Op::AtomicFAddEXT, 16) => {
                            // 16-bit float add: vectors require AtomicFloat16VectorNV,
                            // scalars require AtomicFloat16AddEXT
                            if is_f16_vector {
                                if !ctx.has_capability(Capability::AtomicFloat16VectorNV) {
                                    Some(Capability::AtomicFloat16VectorNV)
                                } else {
                                    None
                                }
                            } else if !ctx.has_capability(Capability::AtomicFloat16AddEXT) {
                                Some(Capability::AtomicFloat16AddEXT)
                            } else {
                                None
                            }
                        }
                        (Op::AtomicFAddEXT, 32) => {
                            if !ctx.has_capability(Capability::AtomicFloat32AddEXT) {
                                Some(Capability::AtomicFloat32AddEXT)
                            } else {
                                None
                            }
                        }
                        (Op::AtomicFAddEXT, 64) => {
                            if !ctx.has_capability(Capability::AtomicFloat64AddEXT) {
                                Some(Capability::AtomicFloat64AddEXT)
                            } else {
                                None
                            }
                        }
                        (Op::AtomicFMinEXT | Op::AtomicFMaxEXT, 16) => {
                            // 16-bit float min/max: vectors require AtomicFloat16VectorNV,
                            // scalars require AtomicFloat16MinMaxEXT
                            if is_f16_vector {
                                if !ctx.has_capability(Capability::AtomicFloat16VectorNV) {
                                    Some(Capability::AtomicFloat16VectorNV)
                                } else {
                                    None
                                }
                            } else if !ctx.has_capability(Capability::AtomicFloat16MinMaxEXT) {
                                Some(Capability::AtomicFloat16MinMaxEXT)
                            } else {
                                None
                            }
                        }
                        (Op::AtomicFMinEXT | Op::AtomicFMaxEXT, 32) => {
                            if !ctx.has_capability(Capability::AtomicFloat32MinMaxEXT) {
                                Some(Capability::AtomicFloat32MinMaxEXT)
                            } else {
                                None
                            }
                        }
                        (Op::AtomicFMinEXT | Op::AtomicFMaxEXT, 64) => {
                            if !ctx.has_capability(Capability::AtomicFloat64MinMaxEXT) {
                                Some(Capability::AtomicFloat64MinMaxEXT)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let Some(cap) = required_capability {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicMissingCapability {
                                function: func,
                                block,
                                opcode: op,
                                required_capability: cap,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Store Pointer Type Rule
// ============================================================================

/// Validates that OpAtomicStore pointer points to integer or float scalar type.
///
/// Per the SPIR-V spec, OpAtomicStore requires the pointer to point to
/// an integer or float scalar type.
pub struct AtomicStorePointerTypeRule;

impl ValidationRule for AtomicStorePointerTypeRule {
    fn name(&self) -> &'static str {
        "atomic-store-pointer-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Only check OpAtomicStore
                    if op != Op::AtomicStore {
                        continue;
                    }

                    // Get pointer operand (first operand)
                    let Some(Operand::IdRef(pointer_id)) = inst.operands.first() else {
                        continue;
                    };

                    // Look up pointer instruction
                    let Some(pointer_rid) = ResultId::try_from(*pointer_id).ok() else {
                        continue;
                    };
                    let Some(pointer_inst) = ctx.definitions.get(&pointer_rid) else {
                        continue;
                    };

                    // Get pointer type
                    let Some(pointer_type_id) = pointer_inst.result_type else {
                        continue;
                    };
                    let Some(pointer_type_rid) = ResultId::try_from(pointer_type_id).ok() else {
                        continue;
                    };
                    let Some(pointer_type) = ctx.definitions.get(&pointer_type_rid) else {
                        continue;
                    };

                    // Get pointee type from OpTypePointer
                    if pointer_type.class.opcode != Op::TypePointer {
                        continue;
                    }

                    let Some(Operand::IdRef(pointee_type_id)) = pointer_type.operands.get(1) else {
                        continue;
                    };

                    // Pointee must be int or float scalar
                    let is_int_scalar = resolver.is_int_scalar(*pointee_type_id, ctx.definitions);
                    let is_float_scalar =
                        resolver.is_float_scalar(*pointee_type_id, ctx.definitions);

                    if !is_int_scalar && !is_float_scalar {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStorePointerNotScalar {
                                function: func,
                                block,
                                opcode: op,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Compare Exchange Comparator Rule
// ============================================================================

/// Validates that OpAtomicCompareExchange/OpAtomicCompareExchangeWeak
/// Comparator operand has the same type as Result Type.
pub struct AtomicCompareExchangeComparatorRule;

impl ValidationRule for AtomicCompareExchangeComparatorRule {
    fn name(&self) -> &'static str {
        "atomic-compare-exchange-comparator"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Only check compare-exchange operations
                    if !matches!(op, Op::AtomicCompareExchange | Op::AtomicCompareExchangeWeak) {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // OpAtomicCompareExchange operands:
                    // 0: Pointer, 1: Memory Scope, 2: Equal semantics, 3: Unequal semantics,
                    // 4: Value, 5: Comparator
                    let Some(Operand::IdRef(comparator_id)) = inst.operands.get(5) else {
                        continue;
                    };

                    // Look up the comparator's type
                    let Some(comparator_rid) = ResultId::try_from(*comparator_id).ok() else {
                        continue;
                    };
                    let Some(comparator_inst) = ctx.definitions.get(&comparator_rid) else {
                        continue;
                    };
                    let Some(comparator_type_id) = comparator_inst.result_type else {
                        continue;
                    };

                    // Comparator type must match Result Type
                    if comparator_type_id != result_type_id {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(
                                ValidationError::AtomicCompareExchangeComparatorTypeMismatch {
                                    function: func,
                                    block,
                                    opcode: op,
                                }.into(),
                        );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Store Value Type Rule
// ============================================================================

/// Validates that OpAtomicStore's Value type matches the Pointer pointee type.
pub struct AtomicStoreValueTypeRule;

impl ValidationRule for AtomicStoreValueTypeRule {
    fn name(&self) -> &'static str {
        "atomic-store-value-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Only check OpAtomicStore
                    if op != Op::AtomicStore {
                        continue;
                    }

                    // OpAtomicStore operands: 0: Pointer, 1: Scope, 2: Semantics, 3: Value
                    // Get pointer operand
                    let Some(Operand::IdRef(pointer_id)) = inst.operands.first() else {
                        continue;
                    };

                    // Get Value operand
                    let Some(Operand::IdRef(value_id)) = inst.operands.get(3) else {
                        continue;
                    };

                    // Look up pointer instruction
                    let Some(pointer_rid) = ResultId::try_from(*pointer_id).ok() else {
                        continue;
                    };
                    let Some(pointer_inst) = ctx.definitions.get(&pointer_rid) else {
                        continue;
                    };

                    // Get pointer type
                    let Some(pointer_type_id) = pointer_inst.result_type else {
                        continue;
                    };
                    let Some(pointer_type_rid) = ResultId::try_from(pointer_type_id).ok() else {
                        continue;
                    };
                    let Some(pointer_type) = ctx.definitions.get(&pointer_type_rid) else {
                        continue;
                    };

                    // Get pointee type from OpTypePointer
                    if pointer_type.class.opcode != Op::TypePointer {
                        continue;
                    }

                    let Some(Operand::IdRef(pointee_type_id)) = pointer_type.operands.get(1) else {
                        continue;
                    };

                    // Get Value type
                    let Some(value_rid) = ResultId::try_from(*value_id).ok() else {
                        continue;
                    };
                    let Some(value_inst) = ctx.definitions.get(&value_rid) else {
                        continue;
                    };
                    let Some(value_type_id) = value_inst.result_type else {
                        continue;
                    };

                    // Value type must match pointee type
                    if value_type_id != *pointee_type_id {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicStoreValueTypeMismatch {
                                function: func,
                                block,
                                opcode: op,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Atomic Value Type Rule
// ============================================================================

/// Validates that atomic operations have Value operand of type Result Type.
///
/// This applies to most atomic operations except:
/// - OpAtomicLoad (no Value operand)
/// - OpAtomicStore (validated separately - Value vs pointee type)
/// - OpAtomicIIncrement (no Value operand)
/// - OpAtomicIDecrement (no Value operand)
/// - OpAtomicFlagTestAndSet (no Value operand)
/// - OpAtomicFlagClear (no Value operand)
pub struct AtomicValueTypeRule;

impl ValidationRule for AtomicValueTypeRule {
    fn name(&self) -> &'static str {
        "atomic-value-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    let op = inst.class.opcode;

                    // Skip operations that don't have a Value operand matching Result Type
                    if matches!(
                        op,
                        Op::AtomicLoad
                            | Op::AtomicStore
                            | Op::AtomicIIncrement
                            | Op::AtomicIDecrement
                            | Op::AtomicFlagTestAndSet
                            | Op::AtomicFlagClear
                    ) {
                        continue;
                    }

                    // Only check atomic operations
                    if !op.is_atomic() {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // For OpAtomicCompareExchange/Weak, Value is at index 4
                    // For other atomics with Value (Exchange, IAdd, ISub, etc), Value is at index 3
                    let value_index = if matches!(
                        op,
                        Op::AtomicCompareExchange | Op::AtomicCompareExchangeWeak
                    ) {
                        4
                    } else {
                        3
                    };

                    let Some(Operand::IdRef(value_id)) = inst.operands.get(value_index) else {
                        continue;
                    };

                    // Look up the Value operand's type
                    let Some(value_rid) = ResultId::try_from(*value_id).ok() else {
                        continue;
                    };
                    let Some(value_inst) = ctx.definitions.get(&value_rid) else {
                        continue;
                    };
                    let Some(value_type_id) = value_inst.result_type else {
                        continue;
                    };

                    // Value type must match Result Type
                    if value_type_id != result_type_id {
                        if let (Some(func), Some(block)) = (function_id, block_id) {
                            return Err(ValidationError::AtomicValueTypeMismatch {
                                function: func,
                                block,
                                opcode: op,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All atomic rules
// ============================================================================

/// Returns all atomic validation rules.
pub fn all_atomic_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &AtomicResultTypeRule,
        &AtomicStorageClassRule,
        &AtomicInt64CapabilityRule,
        &AtomicFloatCapabilityRule,
        &AtomicFlagPointerTypeRule,
        &AtomicStorePointerTypeRule,
        &AtomicCompareExchangeComparatorRule,
        &AtomicStoreValueTypeRule,
        &AtomicValueTypeRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_atomic_op() {
        assert!(Op::AtomicLoad.is_atomic());
        assert!(Op::AtomicStore.is_atomic());
        assert!(Op::AtomicExchange.is_atomic());
        assert!(Op::AtomicIAdd.is_atomic());
        assert!(Op::AtomicFAddEXT.is_atomic());

        assert!(!Op::Load.is_atomic());
        assert!(!Op::Store.is_atomic());
        assert!(!Op::IAdd.is_atomic());
    }

    #[test]
    fn test_has_return_type() {
        assert!(has_return_type(Op::AtomicLoad));
        assert!(has_return_type(Op::AtomicExchange));
        assert!(has_return_type(Op::AtomicIAdd));

        assert!(!has_return_type(Op::AtomicStore));
        assert!(!has_return_type(Op::AtomicFlagClear));
    }

    #[test]
    fn test_requires_float_result() {
        assert!(requires_float_result(Op::AtomicFAddEXT));
        assert!(requires_float_result(Op::AtomicFMinEXT));
        assert!(requires_float_result(Op::AtomicFMaxEXT));

        assert!(!requires_float_result(Op::AtomicIAdd));
        assert!(!requires_float_result(Op::AtomicLoad));
    }

    #[test]
    fn test_requires_int_result() {
        assert!(requires_int_result(Op::AtomicIAdd));
        assert!(requires_int_result(Op::AtomicISub));
        assert!(requires_int_result(Op::AtomicCompareExchange));
        assert!(requires_int_result(Op::AtomicAnd));

        assert!(!requires_int_result(Op::AtomicLoad));
        assert!(!requires_int_result(Op::AtomicFAddEXT));
    }

    #[test]
    fn test_allows_int_or_float_result() {
        assert!(allows_int_or_float_result(Op::AtomicLoad));
        assert!(allows_int_or_float_result(Op::AtomicExchange));

        assert!(!allows_int_or_float_result(Op::AtomicIAdd));
        assert!(!allows_int_or_float_result(Op::AtomicFAddEXT));
    }

    #[test]
    fn test_requires_bool_result() {
        assert!(requires_bool_result(Op::AtomicFlagTestAndSet));

        assert!(!requires_bool_result(Op::AtomicLoad));
        assert!(!requires_bool_result(Op::AtomicIAdd));
    }

    #[test]
    fn test_universal_storage_classes() {
        assert!(StorageClass::Uniform.is_atomic_allowed_universal());
        assert!(StorageClass::Workgroup.is_atomic_allowed_universal());
        assert!(StorageClass::Function.is_atomic_allowed_universal());
        assert!(StorageClass::Image.is_atomic_allowed_universal());

        assert!(!StorageClass::Private.is_atomic_allowed_universal());
        assert!(!StorageClass::Input.is_atomic_allowed_universal());
        assert!(!StorageClass::Output.is_atomic_allowed_universal());
    }

    #[test]
    fn test_vulkan_storage_classes() {
        assert!(StorageClass::Uniform.is_atomic_allowed_vulkan());
        assert!(StorageClass::Workgroup.is_atomic_allowed_vulkan());
        assert!(StorageClass::Image.is_atomic_allowed_vulkan());

        // Function is NOT allowed in Vulkan
        assert!(!StorageClass::Function.is_atomic_allowed_vulkan());
        assert!(!StorageClass::Generic.is_atomic_allowed_vulkan());
    }

    #[test]
    fn test_opencl_storage_classes() {
        assert!(StorageClass::Function.is_atomic_allowed_opencl());
        assert!(StorageClass::Workgroup.is_atomic_allowed_opencl());
        assert!(StorageClass::CrossWorkgroup.is_atomic_allowed_opencl());
        assert!(StorageClass::Generic.is_atomic_allowed_opencl());

        // Uniform is NOT allowed in OpenCL (allowed in Vulkan, not in OpenCL)
        assert!(!StorageClass::Uniform.is_atomic_allowed_opencl());
        assert!(!StorageClass::StorageBuffer.is_atomic_allowed_opencl());
        assert!(!StorageClass::Image.is_atomic_allowed_opencl());
    }

    #[test]
    fn test_is_float16_vec2_or_vec4() {
        use rspirv::dr::Instruction;

        // Create a minimal definitions map with float16, vec2<f16>, vec4<f16>
        let mut definitions: HashMap<ResultId, Instruction> = HashMap::new();

        // Float16 type (id=1)
        let mut float16_inst = Instruction::new(Op::TypeFloat, None, Some(1), vec![]);
        float16_inst
            .operands
            .push(Operand::LiteralBit32(16)); // 16-bit width
        definitions.insert(ResultId::try_from(1).unwrap(), float16_inst);

        // Float32 type (id=2)
        let mut float32_inst = Instruction::new(Op::TypeFloat, None, Some(2), vec![]);
        float32_inst
            .operands
            .push(Operand::LiteralBit32(32)); // 32-bit width
        definitions.insert(ResultId::try_from(2).unwrap(), float32_inst);

        // Vec2<f16> (id=3)
        let mut vec2_f16_inst = Instruction::new(Op::TypeVector, None, Some(3), vec![]);
        vec2_f16_inst.operands.push(Operand::IdRef(1)); // component type = float16
        vec2_f16_inst.operands.push(Operand::LiteralBit32(2)); // count = 2
        definitions.insert(ResultId::try_from(3).unwrap(), vec2_f16_inst);

        // Vec4<f16> (id=4)
        let mut vec4_f16_inst = Instruction::new(Op::TypeVector, None, Some(4), vec![]);
        vec4_f16_inst.operands.push(Operand::IdRef(1)); // component type = float16
        vec4_f16_inst.operands.push(Operand::LiteralBit32(4)); // count = 4
        definitions.insert(ResultId::try_from(4).unwrap(), vec4_f16_inst);

        // Vec3<f16> (id=5) - should NOT match (count != 2 or 4)
        let mut vec3_f16_inst = Instruction::new(Op::TypeVector, None, Some(5), vec![]);
        vec3_f16_inst.operands.push(Operand::IdRef(1)); // component type = float16
        vec3_f16_inst.operands.push(Operand::LiteralBit32(3)); // count = 3
        definitions.insert(ResultId::try_from(5).unwrap(), vec3_f16_inst);

        // Vec2<f32> (id=6) - should NOT match (not float16)
        let mut vec2_f32_inst = Instruction::new(Op::TypeVector, None, Some(6), vec![]);
        vec2_f32_inst.operands.push(Operand::IdRef(2)); // component type = float32
        vec2_f32_inst.operands.push(Operand::LiteralBit32(2)); // count = 2
        definitions.insert(ResultId::try_from(6).unwrap(), vec2_f32_inst);

        // Test valid float16 vec2/vec4
        assert!(is_float16_vec2_or_vec4(3, &definitions), "vec2<f16> should match");
        assert!(is_float16_vec2_or_vec4(4, &definitions), "vec4<f16> should match");

        // Test invalid cases
        assert!(
            !is_float16_vec2_or_vec4(1, &definitions),
            "scalar f16 should not match"
        );
        assert!(
            !is_float16_vec2_or_vec4(5, &definitions),
            "vec3<f16> should not match"
        );
        assert!(
            !is_float16_vec2_or_vec4(6, &definitions),
            "vec2<f32> should not match"
        );
        assert!(
            !is_float16_vec2_or_vec4(999, &definitions),
            "invalid id should not match"
        );
    }
}
