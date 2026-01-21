//! Cooperative matrix and vector type validation rules.
//!
//! This module validates SPIR-V cooperative type requirements:
//! - OpTypeCooperativeMatrixKHR and OpTypeCooperativeMatrixNV requirements
//! - OpTypeCooperativeVectorNV requirements

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

use super::helpers::{get_constant_int_value, is_constant_opcode, is_scalar_numeric_type};

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

impl ValidationRule for TypeCooperativeMatrixRule {
    fn name(&self) -> &'static str {
        "type-cooperative-matrix"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                        }.into());
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
                        }.into());
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
                                        }.into(),
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
                        }.into());
                    }

                    // Try to evaluate and check it's positive
                    if let Some(rows_value) = get_constant_int_value(rows_inst, ctx) {
                        if rows_value <= 0 {
                            return Err(ValidationError::TypeCooperativeMatrixRowsNotPositive {
                                type_id,
                                opcode,
                                value: rows_value,
                            }.into());
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
                        }.into());
                    }

                    // Try to evaluate and check it's positive
                    if let Some(cols_value) = get_constant_int_value(cols_inst, ctx) {
                        if cols_value <= 0 {
                            return Err(ValidationError::TypeCooperativeMatrixColumnsNotPositive {
                                type_id,
                                opcode,
                                value: cols_value,
                            }.into());
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
// OpTypeCooperativeVectorNV Validation Rule
// ============================================================================

/// Validates OpTypeCooperativeVectorNV requirements.
///
/// Checks:
/// - Component Type must be a scalar numerical type (int or float)
/// - Component count must be a constant integer with value >= 1
pub struct TypeCooperativeVectorNVRule;

impl ValidationRule for TypeCooperativeVectorNVRule {
    fn name(&self) -> &'static str {
        "type-cooperative-vector-nv"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::TypeCooperativeVectorNV {
                continue;
            }

            let type_id = inst
                .result_id
                .and_then(|raw| TypeId::try_from(raw).ok())
                .unwrap_or_else(|| TypeId::try_from(0u32).unwrap());

            // Validate Component Type (operand 0) - must be scalar int or float
            let component_type_raw = match inst.operands.first() {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(component_result_id) = ResultId::try_from(component_type_raw) {
                if let Some(component_opcode) = ctx.opcodes.get(&component_result_id) {
                    if !is_scalar_numeric_type(*component_opcode) {
                        let component_type = TypeId::try_from(component_type_raw)
                            .unwrap_or_else(|_| TypeId::try_from(0u32).unwrap());
                        return Err(ValidationError::TypeCooperativeVectorComponentNotScalar {
                            type_id,
                            component_type,
                        }.into());
                    }
                }
            }

            // Validate component count (operand 1) - must be constant integer >= 1
            let count_id_raw = match inst.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                _ => continue,
            };

            if let Ok(count_result_id) = ResultId::try_from(count_id_raw) {
                if let Some(count_inst) = ctx.definitions.get(&count_result_id) {
                    // Must be a constant
                    if !is_constant_opcode(count_inst.class.opcode) {
                        let count_id = Id::try_from(count_id_raw)
                            .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                        return Err(ValidationError::TypeCooperativeVectorCountNotConstant {
                            type_id,
                            count_id,
                        }.into());
                    }

                    // Must have integer type
                    if let Some(count_type_raw) = count_inst.result_type {
                        if let Ok(count_type_result_id) = ResultId::try_from(count_type_raw) {
                            if let Some(count_type_opcode) = ctx.opcodes.get(&count_type_result_id) {
                                if *count_type_opcode != Op::TypeInt {
                                    let count_id = Id::try_from(count_id_raw)
                                        .unwrap_or_else(|_| Id::try_from(1u32).unwrap());
                                    return Err(
                                        ValidationError::TypeCooperativeVectorCountNotInteger {
                                            type_id,
                                            count_id,
                                        }.into(),
                        );
                                }
                            }
                        }
                    }

                    // Check value is >= 1
                    if let Some(count_value) = get_constant_int_value(count_inst, ctx) {
                        if count_value <= 0 {
                            return Err(ValidationError::TypeCooperativeVectorCountInvalid {
                                type_id,
                                value: count_value,
                            }.into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all cooperative type validation rules.
pub fn all_cooperative_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&TypeCooperativeMatrixRule, &TypeCooperativeVectorNVRule]
}
