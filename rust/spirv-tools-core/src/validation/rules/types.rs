//! Type and ID validation rules.
//!
//! This module validates SPIR-V type and ID requirements including:
//!
//! - Result types must be type opcodes
//! - OpTypeFunction parameter validation
//! - Operand definitions
//! - OpTypeInt capability requirements (Int8, Int16, Int64)
//! - OpTypeFloat capability requirements (Float16, Float64)
//! - OpTypeVector component count and capability requirements
//! - OpTypeMatrix column type and count requirements
//! - OpTypeArray/OpTypeRuntimeArray element type requirements

use std::collections::HashSet;

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, Decoration, FPEncoding, Op, StorageClass};

use crate::target_env::TargetEnv;
use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::has_decoration;
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Result Types Are Types Rule
// ============================================================================

/// Validates that result_type fields reference actual type instructions.
pub struct ResultTypesAreTypesRule;

impl ValidationRule for ResultTypesAreTypesRule {
    fn name(&self) -> &'static str {
        "result-types-are-types"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.definitions.values() {
            if let Some(result_type_raw) = inst.result_type {
                if let Ok(type_id) = ResultId::try_from(result_type_raw) {
                    if let Some(type_opcode) = ctx.opcodes.get(&type_id) {
                        if !is_type_opcode(*type_opcode) {
                            return Err(ValidationError::ResultTypeNotType {
                                instruction: inst.class.opcode,
                                result_type: Id::from(type_id),
                                found: *type_opcode,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Type Functions Rule
// ============================================================================

/// Validates OpTypeFunction requirements.
pub struct TypeFunctionsRule;

impl ValidationRule for TypeFunctionsRule {
    fn name(&self) -> &'static str {
        "type-functions"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeFunction {
                continue;
            }
            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .ok_or(ValidationError::ZeroId {
                    kind: crate::validation::types::IdKind::Result,
                    opcode: inst.class.opcode,
                })?;

            let mut operands = inst.operands.iter();
            let return_type = match operands.next() {
                Some(rspirv::dr::Operand::IdRef(raw)) => TypeId::try_from(*raw)
                    .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                _ => {
                    return Err(ValidationError::InvalidTypeFunction { type_id });
                }
            };

            let return_id = ResultId::try_from(u32::from(return_type))
                .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
            let return_opcode = ctx
                .opcodes
                .get(&return_id)
                .copied()
                .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
            if !is_type_opcode(return_opcode) {
                return Err(ValidationError::InvalidTypeFunction { type_id });
            }

            for op in operands {
                let param_type = match op {
                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw)
                        .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                    _ => {
                        return Err(ValidationError::InvalidTypeFunction { type_id });
                    }
                };
                let param_id = ResultId::try_from(u32::from(param_type))
                    .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?;
                let param_opcode = ctx
                    .opcodes
                    .get(&param_id)
                    .copied()
                    .ok_or(ValidationError::InvalidTypeFunction { type_id })?;
                if param_opcode == Op::TypeVoid {
                    return Err(ValidationError::FunctionTypeParameterVoid {
                        type_id,
                        parameter: param_type,
                    });
                }
                if !is_type_opcode(param_opcode) {
                    return Err(ValidationError::InvalidTypeFunction { type_id });
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Operand Definitions Rule
// ============================================================================

/// Validates that all operand IDs are defined.
pub struct OperandDefinitionsRule;

impl ValidationRule for OperandDefinitionsRule {
    fn name(&self) -> &'static str {
        "operand-definitions"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            check_instruction_ids(inst, ctx.defined_ids, None)?;
        }
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|def| def.result_id)
                .and_then(|raw| Id::try_from(raw).ok());
            if let Some(def) = &function.def {
                check_instruction_ids(def, ctx.defined_ids, function_id)?;
            }
            for param in &function.parameters {
                check_instruction_ids(param, ctx.defined_ids, function_id)?;
            }
            for block in &function.blocks {
                for inst in &block.instructions {
                    check_instruction_ids(inst, ctx.defined_ids, function_id)?;
                }
            }
        }
        Ok(())
    }
}

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        return Err(ValidationError::TypeIntRequiresInt8Capability { type_id });
                    }
                }
                16 => {
                    if !has_16bit_int_capability(ctx.declared_capabilities) {
                        return Err(ValidationError::TypeIntRequiresInt16Capability { type_id });
                    }
                }
                32 => {
                    // 32-bit is always valid
                }
                64 => {
                    if !ctx.declared_capabilities.contains(&Capability::Int64) {
                        return Err(ValidationError::TypeIntRequiresInt64Capability { type_id });
                    }
                }
                _ => {
                    return Err(ValidationError::TypeIntInvalidBitWidth { type_id, width });
                }
            }

            // Get signedness (operand 1)
            let signedness = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(s)) => *s,
                _ => continue,
            };

            // Validate signedness value
            if signedness > 1 {
                return Err(ValidationError::TypeIntInvalidSignedness { type_id, signedness });
            }

            // Kernel capability requires signedness 0
            if ctx.declared_capabilities.contains(&Capability::Kernel) && signedness != 0 {
                return Err(ValidationError::TypeIntKernelRequiresUnsigned { type_id });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        });
                    }
                    // 8-bit float requires encoding
                    let Some(enc) = encoding else {
                        return Err(ValidationError::TypeFloat8RequiresEncoding { type_id });
                    };
                    // Only Float8E4M3EXT and Float8E5M2EXT are supported
                    if !matches!(
                        enc,
                        FPEncoding::Float8E4M3EXT | FPEncoding::Float8E5M2EXT
                    ) {
                        return Err(ValidationError::TypeFloat8UnsupportedEncoding {
                            type_id,
                            encoding: enc,
                        });
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
                        });
                    }
                }
                32 => {
                    // 32-bit is always valid
                }
                64 => {
                    if !ctx.declared_capabilities.contains(&Capability::Float64) {
                        return Err(ValidationError::TypeFloatRequiresFloat64Capability {
                            type_id,
                        });
                    }
                }
                _ => {
                    return Err(ValidationError::TypeFloatInvalidBitWidth { type_id, width });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeVector Validation Rule
// ============================================================================

/// Validates OpTypeVector requirements.
///
/// Checks:
/// - Component type must be a scalar type
/// - Component count must be 2, 3, or 4 (or 8, 16 with Vector16 capability)
pub struct TypeVectorRule;

impl ValidationRule for TypeVectorRule {
    fn name(&self) -> &'static str {
        "type-vector"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeVector {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get component type (operand 0)
            let component_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate component type is a scalar
            if let Ok(component_result_id) = ResultId::try_from(component_type_raw) {
                if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                    if !is_scalar_type(*component_opcode) {
                        let component_type = TypeId::try_from(component_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeVectorComponentNotScalar {
                            type_id,
                            component_type,
                        });
                    }
                }
            }

            // Get component count (operand 1)
            let component_count = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(c)) => *c,
                _ => continue,
            };

            // Validate component count
            match component_count {
                2 | 3 | 4 => {
                    // Always valid
                }
                8 | 16 => {
                    if !ctx.declared_capabilities.contains(&Capability::Vector16) {
                        return Err(ValidationError::TypeVectorRequiresVector16Capability {
                            type_id,
                            component_count,
                        });
                    }
                }
                _ => {
                    return Err(ValidationError::TypeVectorInvalidComponentCount {
                        type_id,
                        component_count,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeMatrix Validation Rule
// ============================================================================

/// Validates OpTypeMatrix requirements.
///
/// Checks:
/// - Column type must be a vector type
/// - Vector component type must be a float type
/// - Column count must be 2, 3, or 4
pub struct TypeMatrixRule;

impl ValidationRule for TypeMatrixRule {
    fn name(&self) -> &'static str {
        "type-matrix"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeMatrix {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get column type (operand 0)
            let column_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate column type is a vector
            if let Ok(column_result_id) = ResultId::try_from(column_type_raw) {
                if let Some(column_inst) = ctx.definitions.get(&column_result_id) {
                    if column_inst.class.opcode != Op::TypeVector {
                        return Err(ValidationError::TypeMatrixColumnNotVector { type_id });
                    }

                    // Check that the vector component type is float
                    if let Some(Operand::IdRef(component_type_raw)) = column_inst.operands.first() {
                        if let Ok(component_result_id) = ResultId::try_from(*component_type_raw) {
                            if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                                if *component_opcode != Op::TypeFloat {
                                    return Err(ValidationError::TypeMatrixComponentNotFloat {
                                        type_id,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // Get column count (operand 1)
            let column_count = match inst.operands.get(1) {
                Some(Operand::LiteralBit32(c)) => *c,
                _ => continue,
            };

            // Validate column count
            if column_count < 2 || column_count > 4 {
                return Err(ValidationError::TypeMatrixInvalidColumnCount {
                    type_id,
                    column_count,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeArray Validation Rule
// ============================================================================

/// Validates OpTypeArray requirements.
///
/// Checks:
/// - Element type must not be void
/// - Length must be a constant integer >= 1
pub struct TypeArrayRule;

impl ValidationRule for TypeArrayRule {
    fn name(&self) -> &'static str {
        "type-array"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeArray {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get element type (operand 0)
            let element_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate element type is not void
            if let Ok(element_result_id) = ResultId::try_from(element_type_raw) {
                if let Some(element_opcode) = ctx.opcodes.get(&element_result_id) {
                    if *element_opcode == Op::TypeVoid {
                        return Err(ValidationError::TypeArrayElementVoid { type_id });
                    }
                }
            }

            // Get length (operand 1)
            let length_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate length is a constant
            if let Ok(length_result_id) = ResultId::try_from(length_id_raw) {
                if let Some(length_inst) = ctx.definitions.get(&length_result_id) {
                    if !is_constant_opcode(length_inst.class.opcode) {
                        return Err(ValidationError::TypeArrayLengthNotConstant { type_id });
                    }

                    // Check that the constant type is integer
                    if let Some(const_type_raw) = length_inst.result_type {
                        if let Ok(const_type_result_id) = ResultId::try_from(const_type_raw) {
                            if let Some(const_type_opcode) = ctx.opcodes.get(&const_type_result_id) {
                                if *const_type_opcode != Op::TypeInt {
                                    return Err(ValidationError::TypeArrayLengthNotInteger {
                                        type_id,
                                    });
                                }
                            }
                        }
                    }

                    // Try to evaluate the constant value
                    if let Some(length_value) = get_constant_int_value(length_inst, ctx) {
                        if length_value <= 0 {
                            return Err(ValidationError::TypeArrayLengthInvalid {
                                type_id,
                                length: length_value,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeRuntimeArray Validation Rule
// ============================================================================

/// Validates OpTypeRuntimeArray requirements.
///
/// Checks:
/// - Element type must not be void
pub struct TypeRuntimeArrayRule;

// ============================================================================
// OpTypeCooperativeMatrix Validation Rule
// ============================================================================

/// Validates OpTypeCooperativeMatrixKHR and OpTypeCooperativeMatrixNV requirements.
///
/// Checks:
/// - Component Type must be a scalar integer or floating-point type
/// - Scope must be a constant instruction with scalar integer type
/// - Rows must be a constant with a positive integer value
/// - Columns must be a constant with a positive integer value
/// - Use (KHR only) must be a constant instruction
pub struct TypeCooperativeMatrixRule;

impl ValidationRule for TypeRuntimeArrayRule {
    fn name(&self) -> &'static str {
        "type-runtime-array"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeRuntimeArray {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Get element type (operand 0)
            let element_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            // Validate element type is not void
            if let Ok(element_result_id) = ResultId::try_from(element_type_raw) {
                if let Some(element_opcode) = ctx.opcodes.get(&element_result_id) {
                    if *element_opcode == Op::TypeVoid {
                        return Err(ValidationError::TypeRuntimeArrayElementVoid { type_id });
                    }
                }
            }
        }

        Ok(())
    }
}

impl ValidationRule for TypeCooperativeMatrixRule {
    fn name(&self) -> &'static str {
        "type-cooperative-matrix"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in &ctx.module.types_global_values {
            let opcode = inst.class.opcode;
            if opcode != Op::TypeCooperativeMatrixKHR && opcode != Op::TypeCooperativeMatrixNV {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Operand layout:
            // KHR: Component Type, Scope, Rows, Columns, Use
            // NV:  Component Type, Scope, Rows, Columns

            // Validate Component Type (operand 0) - must be scalar integer or float
            let component_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(component_result_id) = ResultId::try_from(component_type_raw) {
                if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                    if !is_scalar_numeric_type(*component_opcode) {
                        return Err(ValidationError::TypeCooperativeMatrixComponentNotScalar {
                            type_id,
                            opcode,
                        });
                    }
                }
            }

            // Validate Scope (operand 1) - must be constant with scalar integer type
            // Note: rspirv may represent this as IdScope rather than IdRef
            let scope_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                Some(Operand::IdScope(id)) => *id,
                _ => continue,
            };

            if let Ok(scope_result_id) = ResultId::try_from(scope_id_raw) {
                if let Some(scope_inst) = ctx.definitions.get(&scope_result_id) {
                    // Must be a constant instruction
                    if !is_constant_opcode(scope_inst.class.opcode) {
                        return Err(ValidationError::TypeCooperativeMatrixScopeNotConstant {
                            type_id,
                            opcode,
                        });
                    }

                    // Must have integer type
                    if let Some(scope_type_raw) = scope_inst.result_type {
                        if let Ok(scope_type_result_id) = ResultId::try_from(scope_type_raw) {
                            if let Some(scope_type_opcode) = ctx.opcodes.get(&scope_type_result_id) {
                                if *scope_type_opcode != Op::TypeInt {
                                    return Err(
                                        ValidationError::TypeCooperativeMatrixScopeNotInteger {
                                            type_id,
                                            opcode,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Validate Rows (operand 2) - must be constant with positive integer value
            let rows_id_raw = match inst.operands.get(2) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(rows_result_id) = ResultId::try_from(rows_id_raw) {
                if let Some(rows_inst) = ctx.definitions.get(&rows_result_id) {
                    // Must be a constant instruction
                    if !is_constant_opcode(rows_inst.class.opcode) {
                        return Err(ValidationError::TypeCooperativeMatrixRowsNotConstant {
                            type_id,
                            opcode,
                        });
                    }

                    // Try to evaluate and check it's positive
                    if let Some(rows_value) = get_constant_int_value(rows_inst, ctx) {
                        if rows_value <= 0 {
                            return Err(ValidationError::TypeCooperativeMatrixRowsNotPositive {
                                type_id,
                                opcode,
                                value: rows_value,
                            });
                        }
                    }
                }
            }

            // Validate Columns (operand 3) - must be constant with positive integer value
            let cols_id_raw = match inst.operands.get(3) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(cols_result_id) = ResultId::try_from(cols_id_raw) {
                if let Some(cols_inst) = ctx.definitions.get(&cols_result_id) {
                    // Must be a constant instruction
                    if !is_constant_opcode(cols_inst.class.opcode) {
                        return Err(ValidationError::TypeCooperativeMatrixColumnsNotConstant {
                            type_id,
                            opcode,
                        });
                    }

                    // Try to evaluate and check it's positive
                    if let Some(cols_value) = get_constant_int_value(cols_inst, ctx) {
                        if cols_value <= 0 {
                            return Err(ValidationError::TypeCooperativeMatrixColumnsNotPositive {
                                type_id,
                                opcode,
                                value: cols_value,
                            });
                        }
                    }
                }
            }

            // Validate Use (operand 4) - KHR only, must be constant instruction
            if opcode == Op::TypeCooperativeMatrixKHR {
                let use_id_raw = match inst.operands.get(4) {
                    Some(Operand::IdRef(id)) => *id,
                    _ => continue,
                };

                if let Ok(use_result_id) = ResultId::try_from(use_id_raw) {
                    if let Some(use_inst) = ctx.definitions.get(&use_result_id) {
                        // Must be a constant instruction
                        if !is_constant_opcode(use_inst.class.opcode) {
                            return Err(ValidationError::TypeCooperativeMatrixUseNotConstant {
                                type_id,
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpTypeStruct Validation Rule
// ============================================================================

/// Validates OpTypeStruct requirements.
///
/// Checks:
/// - Members cannot be self-references (referring to the struct being defined)
/// - Members must be type instructions
/// - Members cannot be void types
/// - Cannot contain struct members with BuiltIn decoration
/// - (Vulkan) RuntimeArray must be last member and struct must have Block/BufferBlock
/// - Cannot nest Block/BufferBlock decorated structs
/// - BuiltIn decoration must be all-or-nothing for members
/// - (Vulkan) Cannot contain opaque types
pub struct TypeStructRule;

impl ValidationRule for TypeStructRule {
    fn name(&self) -> &'static str {
        "type-struct"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let is_vulkan = matches!(
            ctx.env,
            TargetEnv::Vulkan1_0
                | TargetEnv::Vulkan1_1
                | TargetEnv::Vulkan1_1Spirv1_4
                | TargetEnv::Vulkan1_2
                | TargetEnv::Vulkan1_3
                | TargetEnv::Vulkan1_4
        );

        // Collect struct types with BuiltIn member decorations
        let structs_with_builtin_members = collect_structs_with_builtin_members(ctx);

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeStruct {
                continue;
            }

            let struct_id = inst.result_id.unwrap_or(0);
            let type_id = TypeId::try_from(struct_id)
                .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());

            let member_count = inst.operands.len();

            // Validate each member
            for (member_idx, operand) in inst.operands.iter().enumerate() {
                let member_type_raw = match operand {
                    Operand::IdRef(id) => *id,
                    _ => continue,
                };

                // Check for self-reference
                if member_type_raw == struct_id {
                    return Err(ValidationError::TypeStructMemberSelfReference { type_id });
                }

                // Get the member type instruction
                let member_result_id = match ResultId::try_from(member_type_raw) {
                    Ok(id) => id,
                    Err(_) => continue,
                };

                let member_inst = match ctx.definitions.get(&member_result_id) {
                    Some(inst) => inst,
                    None => continue,
                };

                // Check that member is a type instruction
                if !is_type_opcode(member_inst.class.opcode) {
                    let member_type = TypeId::try_from(member_type_raw)
                        .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                    return Err(ValidationError::TypeStructMemberNotType {
                        type_id,
                        member_type,
                    });
                }

                // Check for void type
                if member_inst.class.opcode == Op::TypeVoid {
                    return Err(ValidationError::TypeStructMemberVoid { type_id });
                }

                // Check for nested struct with BuiltIn members
                if member_inst.class.opcode == Op::TypeStruct {
                    if structs_with_builtin_members.contains(&member_result_id) {
                        let member_type = TypeId::try_from(member_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeStructContainsBuiltInStruct {
                            type_id,
                            member_type,
                        });
                    }
                }

                // Vulkan: RuntimeArray validation
                if is_vulkan && member_inst.class.opcode == Op::TypeRuntimeArray {
                    let is_last_member = member_idx == member_count - 1;
                    if !is_last_member {
                        return Err(ValidationError::TypeStructRuntimeArrayNotLast { type_id });
                    }

                    // Struct must have Block or BufferBlock decoration
                    let has_block =
                        has_decoration(ctx.module, struct_id, Decoration::Block);
                    let has_buffer_block =
                        has_decoration(ctx.module, struct_id, Decoration::BufferBlock);
                    if !has_block && !has_buffer_block {
                        return Err(ValidationError::TypeStructRuntimeArrayNoBlockDecoration {
                            type_id,
                        });
                    }
                }

                // Vulkan: Check for opaque types
                if is_vulkan && !ctx.options.before_hlsl_legalization {
                    if contains_opaque_type(member_type_raw, ctx) {
                        return Err(ValidationError::TypeStructContainsOpaqueType { type_id });
                    }
                }
            }

            // Check for nested Block/BufferBlock
            let this_has_block = has_decoration(ctx.module, struct_id, Decoration::Block);
            let this_has_buffer_block =
                has_decoration(ctx.module, struct_id, Decoration::BufferBlock);

            if this_has_block || this_has_buffer_block {
                if has_nested_block_or_buffer_block(inst, ctx) {
                    return Err(ValidationError::TypeStructNestedBlockOrBufferBlock { type_id });
                }
            }

            // Check BuiltIn all-or-nothing rule
            let builtin_member_count =
                count_builtin_decorated_members(struct_id, ctx);
            if builtin_member_count > 0 && builtin_member_count != member_count {
                return Err(ValidationError::TypeStructBuiltInNotAllMembers {
                    type_id,
                    builtin_count: builtin_member_count,
                    total_count: member_count,
                });
            }
        }

        Ok(())
    }
}

/// Collects all struct type IDs that have at least one member with BuiltIn decoration.
fn collect_structs_with_builtin_members(ctx: &ValidationContext<'_>) -> HashSet<ResultId> {
    let mut result = HashSet::new();

    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(struct_id)),
                Some(Operand::LiteralBit32(_)),
                Some(Operand::Decoration(Decoration::BuiltIn)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if let Ok(result_id) = ResultId::try_from(*struct_id) {
                    result.insert(result_id);
                }
            }
        }
    }

    result
}

/// Counts members of a struct that have BuiltIn decoration.
fn count_builtin_decorated_members(struct_id: u32, ctx: &ValidationContext<'_>) -> usize {
    let mut builtin_members = HashSet::new();

    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(target_id)),
                Some(Operand::LiteralBit32(member_idx)),
                Some(Operand::Decoration(Decoration::BuiltIn)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
            ) {
                if *target_id == struct_id {
                    builtin_members.insert(*member_idx);
                }
            }
        }
    }

    builtin_members.len()
}

/// Checks if a struct has any nested Block or BufferBlock decorated structs.
fn has_nested_block_or_buffer_block(
    struct_inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
) -> bool {
    for operand in &struct_inst.operands {
        let member_type_raw = match operand {
            Operand::IdRef(id) => *id,
            _ => continue,
        };

        if contains_block_or_buffer_block(member_type_raw, ctx, &mut HashSet::new()) {
            return true;
        }
    }
    false
}

/// Recursively checks if a type contains a Block or BufferBlock decorated struct.
fn contains_block_or_buffer_block(
    type_id: u32,
    ctx: &ValidationContext<'_>,
    visited: &mut HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false; // Already visited, prevent infinite recursion
    }

    let result_id = match ResultId::try_from(type_id) {
        Ok(id) => id,
        Err(_) => return false,
    };

    let type_inst = match ctx.definitions.get(&result_id) {
        Some(inst) => inst,
        None => return false,
    };

    match type_inst.class.opcode {
        Op::TypeStruct => {
            // Check if this struct has Block or BufferBlock decoration
            let has_block = has_decoration(ctx.module, type_id, Decoration::Block);
            let has_buffer_block = has_decoration(ctx.module, type_id, Decoration::BufferBlock);
            if has_block || has_buffer_block {
                return true;
            }
            // Check members recursively
            for operand in &type_inst.operands {
                if let Operand::IdRef(member_type_id) = operand {
                    if contains_block_or_buffer_block(*member_type_id, ctx, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            // Check element type
            if let Some(Operand::IdRef(element_type_id)) = type_inst.operands.first() {
                contains_block_or_buffer_block(*element_type_id, ctx, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if a type is or contains an opaque type.
fn contains_opaque_type(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    contains_opaque_type_impl(type_id, ctx, &mut HashSet::new())
}

fn contains_opaque_type_impl(
    type_id: u32,
    ctx: &ValidationContext<'_>,
    visited: &mut HashSet<u32>,
) -> bool {
    if !visited.insert(type_id) {
        return false;
    }

    let result_id = match ResultId::try_from(type_id) {
        Ok(id) => id,
        Err(_) => return false,
    };

    let type_inst = match ctx.definitions.get(&result_id) {
        Some(inst) => inst,
        None => return false,
    };

    // Check if this is an opaque type
    if is_base_opaque_type(type_inst.class.opcode) {
        // Exception: BindlessTextureNV capability allows Image/Sampler/SampledImage
        if ctx.has_capability(Capability::BindlessTextureNV) {
            if matches!(
                type_inst.class.opcode,
                Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage
            ) {
                return false;
            }
        }
        return true;
    }

    // Check nested types
    match type_inst.class.opcode {
        Op::TypeStruct => {
            for operand in &type_inst.operands {
                if let Operand::IdRef(member_type_id) = operand {
                    if contains_opaque_type_impl(*member_type_id, ctx, visited) {
                        return true;
                    }
                }
            }
            false
        }
        Op::TypeArray | Op::TypeRuntimeArray => {
            if let Some(Operand::IdRef(element_type_id)) = type_inst.operands.first() {
                contains_opaque_type_impl(*element_type_id, ctx, visited)
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Checks if an opcode is a base opaque type.
fn is_base_opaque_type(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeOpaque
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
    )
}

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                        });
                    }
                }
            }

            // Validate storage class for target environment
            if !is_valid_storage_class_for_env(storage_class, ctx.env) {
                return Err(ValidationError::TypePointerInvalidStorageClass {
                    type_id,
                    storage_class,
                });
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
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
                return Err(ValidationError::ForwardPointerNotPointerType { target_type });
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
                });
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
                        });
                    }
                }
            }

            // Vulkan: Storage class must be PhysicalStorageBuffer
            if is_vulkan && forward_storage_class != StorageClass::PhysicalStorageBuffer {
                return Err(ValidationError::ForwardPointerRequiresPhysicalStorageBuffer {
                    target_type,
                    storage_class: forward_storage_class,
                });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn is_scalar_type(opcode: Op) -> bool {
    matches!(opcode, Op::TypeBool | Op::TypeInt | Op::TypeFloat)
}

/// Check if opcode is a scalar numeric type (int or float, but not bool).
fn is_scalar_numeric_type(opcode: Op) -> bool {
    matches!(opcode, Op::TypeInt | Op::TypeFloat)
}

fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::ConstantNull
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::SpecConstant
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
    )
}

/// Try to extract a constant integer value from a constant instruction.
fn get_constant_int_value(
    inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
) -> Option<i64> {
    if inst.class.opcode != Op::Constant {
        return None;
    }

    // Get the type to determine signedness and bit width
    let type_id = inst.result_type?;
    let type_result_id = ResultId::try_from(type_id).ok()?;
    let type_inst = ctx.definitions.get(&type_result_id)?;

    if type_inst.class.opcode != Op::TypeInt {
        return None;
    }

    let width = match type_inst.operands.first() {
        Some(Operand::LiteralBit32(w)) => *w,
        _ => return None,
    };

    let signedness = match type_inst.operands.get(1) {
        Some(Operand::LiteralBit32(s)) => *s,
        _ => return None,
    };

    // Get the constant value
    let value = match inst.operands.first() {
        Some(Operand::LiteralBit32(v)) => *v as u64,
        Some(Operand::LiteralBit64(v)) => *v,
        _ => return None,
    };

    // Convert based on signedness and width
    if signedness != 0 {
        // Signed integer
        match width {
            8 => Some(value as i8 as i64),
            16 => Some(value as i16 as i64),
            32 => Some(value as i32 as i64),
            64 => Some(value as i64),
            _ => None,
        }
    } else {
        // Unsigned integer (treat as positive)
        Some(value as i64)
    }
}

fn is_type_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeUntypedPointerKHR
            | Op::TypeFunction
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
            | Op::TypePipeStorage
            | Op::TypeNamedBarrier
            | Op::TypeAccelerationStructureKHR
            | Op::TypeCooperativeMatrixKHR
            | Op::TypeCooperativeMatrixNV
            | Op::TypeRayQueryKHR
            | Op::TypeHitObjectNV
    )
}

#[allow(clippy::manual_is_multiple_of)]
fn is_block_operand(opcode: Op, index: usize) -> bool {
    match opcode {
        Op::Branch => index == 0,
        Op::BranchConditional => index == 1 || index == 2,
        Op::Switch => index == 1 || (index > 1 && index % 2 == 0),
        Op::LoopMerge => index <= 1,
        Op::SelectionMerge => index == 0,
        Op::Phi => index % 2 == 1,
        _ => false,
    }
}

fn check_instruction_ids(
    inst: &rspirv::dr::Instruction,
    defined_ids: &HashSet<Id>,
    function: Option<Id>,
) -> Result<(), ValidationError> {
    if let Some(result_type) = inst.result_type {
        if let Ok(id) = Id::try_from(result_type) {
            if !defined_ids.contains(&id) {
                return Err(ValidationError::UndefinedId { function, id });
            }
        }
    }

    for (idx, operand) in inst.operands.iter().enumerate() {
        if is_block_operand(inst.class.opcode, idx) {
            continue;
        }
        if let rspirv::dr::Operand::IdRef(raw) = operand {
            if let Ok(id) = Id::try_from(*raw) {
                if !defined_ids.contains(&id) {
                    return Err(ValidationError::UndefinedId { function, id });
                }
            }
        }
    }
    Ok(())
}

// ============================================================================
// Type Uniqueness Rule
// ============================================================================

/// Validates that non-aggregate type declarations are unique.
///
/// According to SPIR-V spec section 2.8, non-aggregate types (everything except
/// OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, and OpTypePointer) must be
/// unique. This means you cannot have two OpTypeInt declarations with the same
/// width and signedness, for example.
///
/// Aggregate types are allowed to have multiple identical declarations because
/// they may have different decorations applied.
pub struct TypeUniquenessRule;

/// Checks if the opcode is for an aggregate type that allows duplicate declarations.
fn is_aggregate_type(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeNodePayloadArrayAMDX
            | Op::TypeStruct
            | Op::TypePointer
            | Op::TypeUntypedPointerKHR
    )
}

/// Creates a canonical representation of a type instruction for comparison.
/// Returns None if the type is aggregate (which allows duplicates).
fn type_signature(inst: &rspirv::dr::Instruction) -> Option<(Op, Vec<u32>)> {
    let opcode = inst.class.opcode;

    // Skip non-type opcodes
    if !is_type_opcode(opcode) {
        return None;
    }

    // Skip aggregate types - they're allowed to have duplicates
    if is_aggregate_type(opcode) {
        return None;
    }

    // Create a signature from the opcode and operands
    let mut operand_values = Vec::new();
    for operand in &inst.operands {
        match operand {
            rspirv::dr::Operand::IdRef(id) => operand_values.push(*id),
            rspirv::dr::Operand::LiteralBit32(v) => operand_values.push(*v),
            rspirv::dr::Operand::LiteralBit64(v) => {
                operand_values.push(*v as u32);
                operand_values.push((*v >> 32) as u32);
            }
            rspirv::dr::Operand::Dim(d) => operand_values.push(*d as u32),
            rspirv::dr::Operand::SamplerAddressingMode(m) => operand_values.push(*m as u32),
            rspirv::dr::Operand::SamplerFilterMode(m) => operand_values.push(*m as u32),
            rspirv::dr::Operand::ImageFormat(f) => operand_values.push(*f as u32),
            rspirv::dr::Operand::ImageChannelOrder(o) => operand_values.push(*o as u32),
            rspirv::dr::Operand::ImageChannelDataType(t) => operand_values.push(*t as u32),
            rspirv::dr::Operand::AccessQualifier(q) => operand_values.push(*q as u32),
            rspirv::dr::Operand::StorageClass(sc) => operand_values.push(*sc as u32),
            _ => {} // Skip other operand types for comparison purposes
        }
    }

    Some((opcode, operand_values))
}

impl ValidationRule for TypeUniquenessRule {
    fn name(&self) -> &'static str {
        "type-uniqueness"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        use std::collections::HashMap;

        // Map from type signature to the ID of the first declaration
        let mut seen_types: HashMap<(Op, Vec<u32>), TypeId> = HashMap::new();

        for inst in &ctx.module.types_global_values {
            if let Some(signature) = type_signature(inst) {
                if let Some(result_id) = inst.result_id {
                    if let Ok(type_id) = TypeId::try_from(result_id) {
                        if seen_types.contains_key(&signature) {
                            return Err(ValidationError::DuplicateTypeDeclaration {
                                opcode: inst.class.opcode,
                                type_id,
                            });
                        }
                        seen_types.insert(signature, type_id);
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Reserved Opcode Rule
// ============================================================================

/// Validates that reserved opcodes are not used.
///
/// Certain opcodes are technically defined but are reserved for future use
/// and should never appear in valid SPIR-V modules.
pub struct ReservedOpcodeRule;

impl ValidationRule for ReservedOpcodeRule {
    fn name(&self) -> &'static str {
        "reserved-opcode"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            // These instructions are enabled by a capability but should never be used
            let is_reserved = matches!(
                inst.class.opcode,
                Op::ImageSparseSampleProjImplicitLod
                    | Op::ImageSparseSampleProjExplicitLod
                    | Op::ImageSparseSampleProjDrefImplicitLod
                    | Op::ImageSparseSampleProjDrefExplicitLod
            );

            if is_reserved {
                return Err(ValidationError::ReservedOpcode {
                    opcode: inst.class.opcode,
                });
            }
        }
        Ok(())
    }
}

// ============================================================================
// IdPass Validation Rule
// ============================================================================

/// Validates ID operand usage according to SPIR-V specification.
///
/// This rule implements checks from the C++ IdPass function, including:
/// - Type operands cannot be used where values are expected
/// - Value operands with no type cannot be used where types are expected
/// - Non-semantic instruction results cannot be used in semantic instructions
/// - OpExtInstWithForwardRefsKHR is only allowed with non-semantic instructions
pub struct IdPassRule;

/// Instructions that are allowed to have type operands (without being type-generating).
fn instruction_can_have_type_operand(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::SizeOf
            | Op::CooperativeMatrixLengthNV
            | Op::CooperativeMatrixLengthKHR
            | Op::UntypedArrayLengthKHR
            | Op::Function
            | Op::AsmINTEL
    )
}

/// Instructions that are exempt from the "operand must have a type" requirement.
fn instruction_requires_type_operand(opcode: Op) -> bool {
    // These instructions don't require their operands to have types
    !matches!(
        opcode,
        Op::ExtInst
            | Op::ExtInstWithForwardRefsKHR
            | Op::ExtInstImport
            | Op::SelectionMerge
            | Op::LoopMerge
            | Op::Function
            | Op::SizeOf
            | Op::CooperativeMatrixLengthNV
            | Op::CooperativeMatrixLengthKHR
            | Op::Phi
            | Op::UntypedArrayLengthKHR
            | Op::AsmINTEL
    )
}

/// Checks if an opcode is a debug instruction.
fn is_debug_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::SourceContinued
            | Op::Source
            | Op::SourceExtension
            | Op::Name
            | Op::MemberName
            | Op::String
            | Op::Line
            | Op::NoLine
            | Op::ModuleProcessed
    )
}

/// Checks if an opcode is a decoration instruction.
fn is_decoration_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Decorate
            | Op::MemberDecorate
            | Op::DecorationGroup
            | Op::GroupDecorate
            | Op::GroupMemberDecorate
            | Op::DecorateId
            | Op::DecorateString
            | Op::MemberDecorateString
    )
}

/// Checks if an opcode is a branch instruction.
fn is_branch_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Branch
            | Op::BranchConditional
            | Op::Switch
            | Op::Return
            | Op::ReturnValue
            | Op::Kill
            | Op::Unreachable
            | Op::TerminateInvocation
            | Op::IgnoreIntersectionKHR
            | Op::TerminateRayKHR
    )
}

/// Checks if an opcode generates an untyped pointer (KHR extensions).
fn generates_untyped_pointer(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::UntypedVariableKHR
            | Op::UntypedAccessChainKHR
            | Op::UntypedInBoundsAccessChainKHR
            | Op::UntypedPtrAccessChainKHR
            | Op::UntypedInBoundsPtrAccessChainKHR
    )
}

/// Checks if an instruction is part of a non-semantic extended instruction set.
fn is_non_semantic_instruction(inst: &rspirv::dr::Instruction, ctx: &ValidationContext<'_>) -> bool {
    if inst.class.opcode != Op::ExtInst && inst.class.opcode != Op::ExtInstWithForwardRefsKHR {
        return false;
    }

    // Get the extended instruction set ID (first operand)
    let ext_inst_set_id = match inst.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        _ => return false,
    };

    // Look up the instruction set import to check if it's non-semantic
    for import_inst in &ctx.module.ext_inst_imports {
        if import_inst.result_id == Some(ext_inst_set_id) {
            // Check if the name starts with "NonSemantic."
            if let Some(Operand::LiteralString(name)) = import_inst.operands.first() {
                return name.starts_with("NonSemantic.");
            }
        }
    }
    false
}

impl ValidationRule for IdPassRule {
    fn name(&self) -> &'static str {
        "id-pass"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Validate all instructions in the module
        for inst in ctx.module.all_inst_iter() {
            let opcode = inst.class.opcode;

            // Check if this instruction can have type operands
            let can_have_type_ops = is_type_opcode(opcode)
                || is_debug_opcode(opcode)
                || is_decoration_opcode(opcode)
                || instruction_can_have_type_operand(opcode)
                || generates_untyped_pointer(opcode)
                || is_non_semantic_instruction(inst, ctx);

            // Check if this instruction requires typed operands
            let requires_typed_ops = !is_debug_opcode(opcode)
                && !is_decoration_opcode(opcode)
                && !is_branch_opcode(opcode)
                && !generates_untyped_pointer(opcode)
                && instruction_requires_type_operand(opcode)
                && !is_non_semantic_instruction(inst, ctx);

            // Is this a semantic instruction?
            let is_semantic = !is_non_semantic_instruction(inst, ctx);

            // Check each operand
            for operand in &inst.operands {
                if let Operand::IdRef(operand_id) = operand {
                    let operand_result_id = match ResultId::try_from(*operand_id) {
                        Ok(id) => id,
                        Err(_) => continue,
                    };

                    // Look up the operand's definition
                    if let Some(def_inst) = ctx.definitions.get(&operand_result_id) {
                        let def_opcode = def_inst.class.opcode;

                        // Check: Type operand cannot be used where value is expected
                        if is_type_opcode(def_opcode) && !can_have_type_ops {
                            return Err(ValidationError::OperandCannotBeType {
                                operand: Id::try_from(*operand_id).unwrap_or(Id::try_from(1u32).unwrap()),
                            });
                        }

                        // Check: Operand must have a type if required
                        if def_inst.result_type.is_none()
                            && !is_type_opcode(def_opcode)
                            && requires_typed_ops
                        {
                            return Err(ValidationError::OperandRequiresType {
                                operand: Id::try_from(*operand_id).unwrap_or(Id::try_from(1u32).unwrap()),
                            });
                        }

                        // Check: Non-semantic result cannot be used in semantic instruction
                        if is_semantic && is_non_semantic_instruction(def_inst, ctx) {
                            return Err(ValidationError::NonSemanticUsedInSemantic {
                                operand: Id::try_from(*operand_id).unwrap_or(Id::try_from(1u32).unwrap()),
                            });
                        }
                    }
                }
            }

            // Check OpExtInstWithForwardRefsKHR specific requirements
            if opcode == Op::ExtInstWithForwardRefsKHR {
                // Must be a non-semantic instruction
                if !is_non_semantic_instruction(inst, ctx) {
                    return Err(ValidationError::ExtInstWithForwardRefsNotNonSemantic);
                }

                // Must have at least one forward reference
                // (This is hard to check without tracking forward declarations,
                // so we'll skip this for now - the C++ version tracks this during parsing)
            }
        }

        Ok(())
    }
}

// ============================================================================
// All type rules
// ============================================================================

/// Returns all type validation rules.
pub fn all_type_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ReservedOpcodeRule,
        &TypeUniquenessRule,
        &ResultTypesAreTypesRule,
        &TypeFunctionsRule,
        &OperandDefinitionsRule,
        &TypeIntRule,
        &TypeFloatRule,
        &TypeVectorRule,
        &TypeMatrixRule,
        &TypeArrayRule,
        &TypeRuntimeArrayRule,
        &TypeCooperativeMatrixRule,
        &TypeStructRule,
        &TypePointerRule,
        &TypeForwardPointerRule,
        &IdPassRule,
    ]
}
