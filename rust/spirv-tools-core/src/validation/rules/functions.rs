//! Function instruction validation rules.
//!
//! This module validates SPIR-V function-related instructions:
//!
//! - OpFunction: Function definition validation
//! - OpFunctionParameter: Parameter type matching
//! - OpFunctionCall: Call signature matching and argument validation
//!
//! Function validation ensures proper function type matching,
//! parameter counts and types, and proper function usage.

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};

// ============================================================================
// Function Definition Rule
// ============================================================================

/// Validates OpFunction instructions.
///
/// Ensures that:
/// - Function Type is actually OpTypeFunction
/// - Result type matches the function type's return type
pub struct FunctionDefinitionRule;

impl ValidationRule for FunctionDefinitionRule {
    fn name(&self) -> &'static str {
        "function-definition"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let Some(def) = &function.def else {
                continue;
            };

            let function_id = def.result_id.and_then(|id| Id::try_from(id).ok());

            // Get the function type operand (operand 1 in OpFunction)
            let function_type_id = def.operands.get(1).and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });

            let Some(function_type_id) = function_type_id else {
                continue;
            };

            // Check that it's actually OpTypeFunction
            let function_type_inst = ResultId::try_from(function_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            let Some(function_type_inst) = function_type_inst else {
                // Reference to undefined type - other validation will catch this
                continue;
            };

            if function_type_inst.class.opcode != Op::TypeFunction {
                if let (Some(func_id), Ok(type_id)) =
                    (function_id, TypeId::try_from(function_type_id))
                {
                    return Err(ValidationError::FunctionTypeInvalid {
                        function: func_id,
                        function_type: type_id,
                        expected: "OpTypeFunction",
                    });
                }
            }

            // Check that the return type matches
            let declared_return_type = def.result_type;
            let function_return_type = function_type_inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });

            if let (Some(declared), Some(expected)) = (declared_return_type, function_return_type) {
                if declared != expected {
                    if let (Some(func_id), Ok(result_type), Ok(func_type)) = (
                        function_id,
                        TypeId::try_from(declared),
                        TypeId::try_from(expected),
                    ) {
                        return Err(ValidationError::FunctionReturnTypeMismatch {
                            function: func_id,
                            result_type,
                            function_type: func_type,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Function Parameter Rule
// ============================================================================

/// Validates OpFunctionParameter instructions.
///
/// Ensures that:
/// - Parameter type matches the corresponding type in OpTypeFunction
/// - Not more parameters than declared in the function type
pub struct FunctionParameterRule;

impl ValidationRule for FunctionParameterRule {
    fn name(&self) -> &'static str {
        "function-parameter"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let Some(def) = &function.def else {
                continue;
            };

            let function_id = def.result_id.and_then(|id| Id::try_from(id).ok());

            // Get the function type
            let function_type_id = def.operands.get(1).and_then(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });

            let function_type_inst = function_type_id
                .and_then(|id| ResultId::try_from(id).ok())
                .and_then(|rid| ctx.definitions.get(&rid));

            let Some(function_type_inst) = function_type_inst else {
                continue;
            };

            if function_type_inst.class.opcode != Op::TypeFunction {
                continue;
            }

            // Get the expected parameter types from OpTypeFunction
            // Operands: [return_type, param1_type, param2_type, ...]
            let expected_param_count = function_type_inst.operands.len().saturating_sub(1);

            // Check parameter count
            if function.parameters.len() != expected_param_count {
                if let Some(func_id) = function_id {
                    return Err(ValidationError::FunctionParameterCountMismatch {
                        function: func_id,
                        expected: expected_param_count,
                        found: function.parameters.len(),
                    });
                }
            }

            // Check each parameter's type
            for (param_idx, param) in function.parameters.iter().enumerate() {
                // Get the expected type for this parameter
                let expected_type_id =
                    function_type_inst
                        .operands
                        .get(param_idx + 1)
                        .and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        });

                let actual_type_id = param.result_type;

                if let (Some(expected), Some(actual)) = (expected_type_id, actual_type_id) {
                    if expected != actual {
                        let param_id = param.result_id.and_then(|id| Id::try_from(id).ok());
                        if let (Some(func_id), Some(param), Ok(expected_type), Ok(actual_type)) = (
                            function_id,
                            param_id,
                            TypeId::try_from(expected),
                            TypeId::try_from(actual),
                        ) {
                            return Err(ValidationError::FunctionParameterTypeMismatch {
                                function: func_id,
                                parameter: param,
                                expected: expected_type,
                                found: actual_type,
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
// Function Call Rule
// ============================================================================

/// Validates OpFunctionCall instructions.
///
/// Ensures that:
/// - Callee is actually a function
/// - Return type matches
/// - Argument count matches parameter count
/// - Argument types match parameter types
pub struct FunctionCallRule;

impl ValidationRule for FunctionCallRule {
    fn name(&self) -> &'static str {
        "function-call"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode != Op::FunctionCall {
                        continue;
                    }

                    let Some(func_id) = function_id else {
                        continue;
                    };

                    // Get the callee function ID (first operand)
                    let callee_id = inst.operands.first().and_then(|op| match op {
                        Operand::IdRef(id) => Some(*id),
                        _ => None,
                    });

                    let Some(callee_id) = callee_id else {
                        continue;
                    };

                    // Check that callee is a function
                    let callee_opcode = ResultId::try_from(callee_id)
                        .ok()
                        .and_then(|rid| ctx.opcodes.get(&rid))
                        .copied();

                    if let Some(opcode) = callee_opcode {
                        if opcode != Op::Function {
                            if let Ok(target) = Id::try_from(callee_id) {
                                return Err(ValidationError::FunctionCallTargetNotFunction {
                                    function: func_id,
                                    target,
                                    found: opcode,
                                });
                            }
                        }
                    }

                    // Get the callee's function definition
                    let callee_def = ResultId::try_from(callee_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(callee_def) = callee_def else {
                        continue;
                    };

                    // Return type check: call's result type should match callee's result type
                    if let (Some(call_result_type), Some(callee_result_type)) =
                        (inst.result_type, callee_def.result_type)
                    {
                        if call_result_type != callee_result_type {
                            if let (Ok(found), Ok(expected)) = (
                                TypeId::try_from(call_result_type),
                                TypeId::try_from(callee_result_type),
                            ) {
                                return Err(ValidationError::FunctionCallResultTypeMismatch {
                                    function: func_id,
                                    expected,
                                    found,
                                });
                            }
                        }
                    }

                    // Get the function type from callee
                    let callee_func_type_id =
                        callee_def.operands.get(1).and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        });

                    let callee_func_type = callee_func_type_id
                        .and_then(|id| ResultId::try_from(id).ok())
                        .and_then(|rid| ctx.definitions.get(&rid));

                    let Some(callee_func_type) = callee_func_type else {
                        continue;
                    };

                    // Count arguments in call (operands after the function ID)
                    let argument_count = inst.operands.len().saturating_sub(1);
                    // Parameter count from function type (operands after return type)
                    let parameter_count = callee_func_type.operands.len().saturating_sub(1);

                    if argument_count != parameter_count {
                        return Err(ValidationError::FunctionCallArgumentCountMismatch {
                            function: func_id,
                            expected: parameter_count,
                            found: argument_count,
                        });
                    }

                    // Check argument types match parameter types
                    for (arg_idx, arg_op) in inst.operands.iter().skip(1).enumerate() {
                        let arg_id = match arg_op {
                            Operand::IdRef(id) => *id,
                            _ => continue,
                        };

                        let arg_type = ResultId::try_from(arg_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                            .and_then(|inst| inst.result_type);

                        let param_type =
                            callee_func_type
                                .operands
                                .get(arg_idx + 1)
                                .and_then(|op| match op {
                                    Operand::IdRef(id) => Some(*id),
                                    _ => None,
                                });

                        if let (Some(arg_type), Some(param_type)) = (arg_type, param_type) {
                            if arg_type != param_type {
                                if let (Ok(argument), Ok(found), Ok(expected)) = (
                                    Id::try_from(arg_id),
                                    TypeId::try_from(arg_type),
                                    TypeId::try_from(param_type),
                                ) {
                                    return Err(ValidationError::FunctionCallArgumentTypeMismatch {
                                        function: func_id,
                                        argument,
                                        expected,
                                        found,
                                    });
                                }
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
// All function rules
// ============================================================================

/// Returns all function validation rules.
pub fn all_function_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &FunctionDefinitionRule,
        &FunctionParameterRule,
        &FunctionCallRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_rules_exist() {
        let rules = all_function_rules();
        assert_eq!(rules.len(), 3);
        assert_eq!(rules[0].name(), "function-definition");
        assert_eq!(rules[1].name(), "function-parameter");
        assert_eq!(rules[2].name(), "function-call");
    }
}
