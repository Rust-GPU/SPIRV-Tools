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
use rspirv::spirv::{FunctionControl, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId, TypeId};
use crate::validation::ValidationResult;

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    }
                    .into());
                }
            }

            // Check that the return type matches
            let declared_return_type = def.result_type;
            let function_return_type =
                function_type_inst.operands.first().and_then(|op| match op {
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
                        }
                        .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    }
                    .into());
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
                            }
                            .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                }
                                .into());
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
                                }
                                .into());
                            }
                        }
                    }

                    // Get the function type from callee
                    let callee_func_type_id = callee_def.operands.get(1).and_then(|op| match op {
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
                        }
                        .into());
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
                                    return Err(
                                        ValidationError::FunctionCallArgumentTypeMismatch {
                                            function: func_id,
                                            argument,
                                            expected,
                                            found,
                                        }
                                        .into(),
                                    );
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
// Return Value Rule
// ============================================================================

/// Validates OpReturn and OpReturnValue instructions.
///
/// Ensures that:
/// - OpReturn is only used in void-returning functions
/// - OpReturnValue is only used in non-void functions
/// - OpReturnValue type matches the function's return type
pub struct ReturnValueRule;

impl ValidationRule for ReturnValueRule {
    fn name(&self) -> &'static str {
        "return-value"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let Some(def) = &function.def else {
                continue;
            };

            let function_id = def
                .result_id
                .and_then(|id| Id::try_from(id).ok())
                .unwrap_or_else(|| Id::try_from(1u32).unwrap());

            // Get the return type from the function definition
            let return_type_id = def.result_type;
            let Some(return_type_id) = return_type_id else {
                continue;
            };

            // Check if the return type is void
            let return_type_inst = ResultId::try_from(return_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid));

            let is_void = return_type_inst
                .map(|inst| inst.class.opcode == Op::TypeVoid)
                .unwrap_or(false);

            let return_type = TypeId::try_from(return_type_id).ok();

            for block in &function.blocks {
                for inst in &block.instructions {
                    match inst.class.opcode {
                        Op::Return => {
                            if !is_void {
                                if let Some(expected) = return_type {
                                    return Err(ValidationError::MissingReturnValue {
                                        function: function_id,
                                        expected,
                                    }
                                    .into());
                                }
                            }
                        }
                        Op::ReturnValue => {
                            if is_void {
                                return Err(ValidationError::ReturnValueInVoidFunction {
                                    function: function_id,
                                }
                                .into());
                            }

                            // Check that the returned value has the correct type
                            if let Some(Operand::IdRef(value_id)) = inst.operands.first() {
                                if let Ok(value_rid) = ResultId::try_from(*value_id) {
                                    if let Some(value_type) = ctx.result_types.get(&value_rid) {
                                        if let Some(expected) = return_type {
                                            if *value_type != expected {
                                                return Err(
                                                    ValidationError::InvalidReturnValueType {
                                                        function: function_id,
                                                        expected,
                                                        found: *value_type,
                                                    }
                                                    .into(),
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
            }
        }

        Ok(())
    }
}

// ============================================================================
// Function Variable Rule
// ============================================================================

/// Validates OpVariable instructions within functions.
///
/// Ensures that:
/// - Variables in functions must have Function storage class
/// - Variables must be declared in the entry block
pub struct FunctionVariableRule;

impl ValidationRule for FunctionVariableRule {
    fn name(&self) -> &'static str {
        "function-variable"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok())
                .unwrap_or_else(|| Id::try_from(1u32).unwrap());

            for (block_index, block) in function.blocks.iter().enumerate() {
                for inst in &block.instructions {
                    if inst.class.opcode == Op::Variable {
                        // Get the storage class
                        let storage_class = inst.operands.first().and_then(|op| {
                            if let Operand::StorageClass(sc) = op {
                                Some(*sc)
                            } else {
                                None
                            }
                        });

                        let variable_id = inst
                            .result_id
                            .and_then(|id| Id::try_from(id).ok())
                            .unwrap_or(function_id);

                        // Check storage class is Function
                        if let Some(sc) = storage_class {
                            if sc != rspirv::spirv::StorageClass::Function {
                                return Err(
                                    ValidationError::FunctionVariableStorageClassMismatch {
                                        function: function_id,
                                        variable: variable_id,
                                        storage_class: sc,
                                    }
                                    .into(),
                                );
                            }
                        }

                        // Check variable is in entry block
                        if block_index != 0 {
                            return Err(ValidationError::FunctionVariableNotInEntryBlock {
                                function: function_id,
                                variable: variable_id,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Function Use Rule
// ============================================================================

/// Validates that function result IDs are only used in acceptable contexts.
///
/// From the SPIR-V spec, function IDs can only be used in specific operations:
/// - OpGroupDecorate, OpDecorate (decorating the function)
/// - OpEnqueueKernel
/// - OpEntryPoint
/// - OpExecutionMode, OpExecutionModeId
/// - OpFunctionCall
/// - OpGetKernelNDrangeSubGroupCount, OpGetKernelNDrangeMaxSubGroupSize
/// - OpGetKernelWorkGroupSize, OpGetKernelPreferredWorkGroupSizeMultiple
/// - OpGetKernelLocalSizeForSubgroupCount, OpGetKernelMaxNumSubgroups
/// - OpName
/// - OpCooperativeMatrixPerElementOpNV, OpCooperativeMatrixReduceNV
/// - OpCooperativeMatrixLoadTensorNV
/// - OpConditionalEntryPointINTEL
/// - NonSemantic/Debug instructions
pub struct FunctionUseRule;

impl FunctionUseRule {
    /// Returns true if the opcode is an acceptable use of a function result ID.
    fn is_acceptable_function_use(opcode: Op) -> bool {
        matches!(
            opcode,
            Op::GroupDecorate
                | Op::Decorate
                | Op::EnqueueKernel
                | Op::EntryPoint
                | Op::ExecutionMode
                | Op::ExecutionModeId
                | Op::FunctionCall
                | Op::GetKernelNDrangeSubGroupCount
                | Op::GetKernelNDrangeMaxSubGroupSize
                | Op::GetKernelWorkGroupSize
                | Op::GetKernelPreferredWorkGroupSizeMultiple
                | Op::GetKernelLocalSizeForSubgroupCount
                | Op::GetKernelMaxNumSubgroups
                | Op::Name
                | Op::CooperativeMatrixPerElementOpNV
                | Op::CooperativeMatrixReduceNV
                | Op::CooperativeMatrixLoadTensorNV
                | Op::ConditionalEntryPointINTEL
        )
    }

    /// Returns true if the instruction is a NonSemantic or Debug instruction.
    fn is_non_semantic_or_debug(opcode: Op) -> bool {
        // NonSemantic instructions start at 5000+ range
        // Debug instructions include various debug ops
        matches!(
            opcode,
            Op::ExtInst | Op::Line | Op::NoLine | Op::ModuleProcessed
        )
    }
}

impl ValidationRule for FunctionUseRule {
    fn name(&self) -> &'static str {
        "function-use"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Collect all function IDs
        let mut function_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for function in &ctx.module.functions {
            if let Some(def) = &function.def {
                if let Some(id) = def.result_id {
                    function_ids.insert(id);
                }
            }
        }

        // Check all instructions for uses of function IDs
        for inst in ctx.module.all_inst_iter() {
            let opcode = inst.class.opcode;

            // Skip if this is an acceptable use of functions
            if Self::is_acceptable_function_use(opcode) || Self::is_non_semantic_or_debug(opcode) {
                continue;
            }

            // Check each operand to see if it references a function
            for operand in &inst.operands {
                if let Operand::IdRef(id) = operand {
                    if function_ids.contains(id) {
                        if let Ok(func_id) = Id::try_from(*id) {
                            return Err(ValidationError::FunctionInvalidUse {
                                function: func_id,
                                use_opcode: opcode,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Function Declaration Order Rule
// ============================================================================

/// Validates that function declarations come before definitions.
///
/// In SPIR-V, function declarations (functions with no blocks) must all
/// appear before any function definitions (functions with blocks).
pub struct FunctionDeclarationOrderRule;

impl ValidationRule for FunctionDeclarationOrderRule {
    fn name(&self) -> &'static str {
        "function-declaration-order"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let mut seen_definition = false;

        for function in &ctx.module.functions {
            let is_declaration = function.blocks.is_empty() && function.parameters.is_empty();

            if is_declaration {
                if seen_definition {
                    let function_id = function
                        .def
                        .as_ref()
                        .and_then(|d| d.result_id)
                        .and_then(|id| Id::try_from(id).ok())
                        .unwrap_or_else(|| Id::try_from(1u32).unwrap());

                    return Err(ValidationError::FunctionDeclarationAfterDefinition {
                        function: function_id,
                    }
                    .into());
                }
            } else {
                seen_definition = true;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Function Control Rule
// ============================================================================

/// Validates function control flags in OpFunction.
///
/// Checks:
/// - Inline and DontInline cannot both be specified
pub struct FunctionControlRule;

impl ValidationRule for FunctionControlRule {
    fn name(&self) -> &'static str {
        "function-control"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for function in &ctx.module.functions {
            let Some(def) = &function.def else {
                continue;
            };

            let function_id = def
                .result_id
                .and_then(|id| Id::try_from(id).ok())
                .unwrap_or_else(|| Id::try_from(1u32).unwrap());

            // Get function control (operand 0 in OpFunction)
            let function_control = def.operands.first().and_then(|op| match op {
                Operand::FunctionControl(ctrl) => Some(*ctrl),
                _ => None,
            });

            let Some(function_control) = function_control else {
                continue;
            };

            // Check Inline and DontInline cannot both be specified
            if function_control.contains(FunctionControl::INLINE)
                && function_control.contains(FunctionControl::DONT_INLINE)
            {
                return Err(ValidationError::FunctionControlInlineAndDontInline {
                    function: function_id,
                }
                .into());
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
        &ReturnValueRule,
        &FunctionVariableRule,
        &FunctionUseRule,
        &FunctionDeclarationOrderRule,
        &FunctionControlRule,
    ]
}
