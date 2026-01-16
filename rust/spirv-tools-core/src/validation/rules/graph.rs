//! ARM graph instruction validation rules.
//!
//! This module validates SPIR-V ARM graph instructions including:
//!
//! - OpTypeGraphARM - graph type definition
//! - OpGraphConstantARM - graph constant definition
//! - OpGraphEntryPointARM - graph entry point declaration
//! - OpGraphARM - graph definition start
//! - OpGraphInputARM - graph input access
//! - OpGraphSetOutputARM - graph output setting
//! - OpGraphEndARM - graph definition end

use std::collections::{HashMap, HashSet};

use rspirv::dr::Operand;
use rspirv::spirv::{Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a type is a tensor type (OpTypeTensorARM).
fn is_tensor_type(type_id: u32, definitions: &HashMap<ResultId, rspirv::dr::Instruction>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            return inst.class.opcode == Op::TypeTensorARM;
        }
    }
    false
}

/// Check if a type is an array of tensors.
fn is_tensor_array(type_id: u32, definitions: &HashMap<ResultId, rspirv::dr::Instruction>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(inst) = definitions.get(&result_id) {
            if inst.class.opcode == Op::TypeArray || inst.class.opcode == Op::TypeRuntimeArray {
                // Get element type (word 2 for arrays)
                if let Some(Operand::IdRef(elem_type_id)) = inst.operands.first() {
                    return is_tensor_type(*elem_type_id, definitions);
                }
            }
        }
    }
    false
}

/// Check if a type is a valid graph interface type (tensor or tensor array).
fn is_graph_interface_type(
    type_id: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> bool {
    is_tensor_type(type_id, definitions) || is_tensor_array(type_id, definitions)
}

/// Check if an instruction is an OpGraphARM.
fn is_graph_inst(inst: &rspirv::dr::Instruction) -> bool {
    inst.class.opcode == Op::GraphARM
}

/// Check if an instruction is an OpTypeGraphARM.
fn is_graph_type(inst: &rspirv::dr::Instruction) -> bool {
    inst.class.opcode == Op::TypeGraphARM
}

/// Get the number of inputs from an OpTypeGraphARM instruction.
fn graph_type_num_inputs(inst: &rspirv::dr::Instruction) -> u32 {
    // OpTypeGraphARM: result_id, NumInputs, IO types...
    if let Some(Operand::LiteralBit32(num_inputs)) = inst.operands.first() {
        *num_inputs
    } else {
        0
    }
}

/// Get the total number of I/O types from an OpTypeGraphARM instruction.
fn graph_type_num_io(inst: &rspirv::dr::Instruction) -> usize {
    // Operands after NumInputs are I/O types
    inst.operands.len().saturating_sub(1)
}

/// Get the number of outputs from an OpTypeGraphARM instruction.
fn graph_type_num_outputs(inst: &rspirv::dr::Instruction) -> usize {
    graph_type_num_io(inst).saturating_sub(graph_type_num_inputs(inst) as usize)
}

/// Get an input type ID at the given index from an OpTypeGraphARM.
fn graph_type_input_at(inst: &rspirv::dr::Instruction, index: usize) -> Option<u32> {
    // Operands: NumInputs, IO_type_0, IO_type_1, ...
    inst.operands.get(1 + index).and_then(|op| match op {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    })
}

/// Get an output type ID at the given index from an OpTypeGraphARM.
fn graph_type_output_at(inst: &rspirv::dr::Instruction, index: usize) -> Option<u32> {
    let num_inputs = graph_type_num_inputs(inst) as usize;
    inst.operands.get(1 + num_inputs + index).and_then(|op| match op {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    })
}

/// Try to evaluate a constant u64 value from an ID.
fn eval_constant_u64(
    id: u32,
    definitions: &HashMap<ResultId, rspirv::dr::Instruction>,
) -> Option<u64> {
    let result_id = ResultId::try_from(id).ok()?;
    let inst = definitions.get(&result_id)?;
    if inst.class.opcode == Op::Constant {
        if let Some(Operand::LiteralBit32(val)) = inst.operands.first() {
            return Some(*val as u64);
        }
        if let Some(Operand::LiteralBit64(val)) = inst.operands.first() {
            return Some(*val);
        }
    }
    None
}

/// Validates OpTypeGraphARM instructions.
///
/// - Must have at least NumInputs types
/// - Must have at least one output
/// - All I/O types must be graph interface types (tensor or tensor array)
pub struct GraphTypeRule;

impl ValidationRule for GraphTypeRule {
    fn name(&self) -> &'static str {
        "graph-type"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::TypeGraphARM {
                continue;
            }

            let num_inputs = graph_type_num_inputs(inst);
            let num_io = graph_type_num_io(inst);

            // Check there are at least NumInputs types
            if num_io < num_inputs as usize {
                return Err(ValidationError::GraphTypeTooFewIOTypes {
                    instruction_id: inst.result_id.map(to_id),
                    num_io_types: num_io,
                    num_inputs,
                });
            }

            // Check there is at least one output
            if num_io == num_inputs as usize {
                return Err(ValidationError::GraphTypeNoOutputs {
                    instruction_id: inst.result_id.map(to_id),
                });
            }

            // Check all I/O types are graph interface types
            for i in 1..=num_io {
                if let Some(Operand::IdRef(type_id)) = inst.operands.get(i) {
                    if !is_graph_interface_type(*type_id, ctx.definitions) {
                        return Err(ValidationError::GraphTypeInvalidIOType {
                            instruction_id: inst.result_id.map(to_id),
                            io_type: to_id(*type_id),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpGraphConstantARM instructions.
///
/// - Result type must be a tensor type
/// - No two OpGraphConstantARM instructions may have the same GraphConstantID
pub struct GraphConstantRule;

impl ValidationRule for GraphConstantRule {
    fn name(&self) -> &'static str {
        "graph-constant"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let mut seen_constant_ids: HashSet<u32> = HashSet::new();

        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::GraphConstantARM {
                continue;
            }

            // Check result type is a tensor type
            if let Some(result_type_id) = inst.result_type {
                if !is_tensor_type(result_type_id, ctx.definitions) {
                    return Err(ValidationError::GraphConstantNotTensorType {
                        instruction_id: inst.result_id.map(to_id),
                    });
                }
            }

            // Check for duplicate GraphConstantID (operand 0)
            if let Some(Operand::LiteralBit32(constant_id)) = inst.operands.first() {
                if !seen_constant_ids.insert(*constant_id) {
                    return Err(ValidationError::GraphConstantDuplicateId {
                        instruction_id: inst.result_id.map(to_id),
                        constant_id: *constant_id,
                    });
                }
            }
        }

        Ok(())
    }
}

/// Validates OpGraphARM instructions.
///
/// - Result type must be an OpTypeGraphARM
pub struct GraphRule;

impl ValidationRule for GraphRule {
    fn name(&self) -> &'static str {
        "graph"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::GraphARM {
                continue;
            }

            // Check result type is OpTypeGraphARM
            if let Some(result_type_id) = inst.result_type {
                if let Ok(type_result_id) = ResultId::try_from(result_type_id) {
                    if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                        if !is_graph_type(type_inst) {
                            return Err(ValidationError::GraphInvalidResultType {
                                instruction_id: inst.result_id.map(to_id),
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpGraphEntryPointARM instructions.
///
/// - Graph operand must be an OpGraphARM
/// - Number of interface IDs must match the graph type's I/O count
/// - Interface IDs must come from OpVariable with UniformConstant storage class
/// - Interface type must match corresponding graph I/O type
pub struct GraphEntryPointRule;

impl ValidationRule for GraphEntryPointRule {
    fn name(&self) -> &'static str {
        "graph-entry-point"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::GraphEntryPointARM {
                continue;
            }

            // Check Graph operand is an OpGraphARM
            let Some(Operand::IdRef(graph_id)) = inst.operands.first() else {
                continue;
            };

            let Some(graph_inst) = ResultId::try_from(*graph_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid))
            else {
                continue;
            };

            if !is_graph_inst(graph_inst) {
                return Err(ValidationError::GraphEntryPointInvalidGraph {
                    instruction_id: inst.result_id.map(to_id),
                    graph_id: to_id(*graph_id),
                });
            }

            // Get graph type
            let Some(graph_type_id) = graph_inst.result_type else {
                continue;
            };
            let Some(graph_type_inst) = ResultId::try_from(graph_type_id)
                .ok()
                .and_then(|rid| ctx.definitions.get(&rid))
            else {
                continue;
            };

            if !is_graph_type(graph_type_inst) {
                // Invalid type, but let GraphRule report that
                continue;
            }

            // Check interface count matches I/O count
            let num_io = graph_type_num_io(graph_type_inst);
            let num_interface = inst.operands.len().saturating_sub(2); // Skip Graph, Name

            if num_io != num_interface {
                return Err(ValidationError::GraphEntryPointInterfaceCountMismatch {
                    instruction_id: inst.result_id.map(to_id),
                    expected: num_io,
                    actual: num_interface,
                });
            }

            // Check each interface variable
            for (i, operand) in inst.operands.iter().skip(2).enumerate() {
                let Operand::IdRef(interface_id) = operand else {
                    continue;
                };

                let Some(interface_inst) = ResultId::try_from(*interface_id)
                    .ok()
                    .and_then(|rid| ctx.definitions.get(&rid))
                else {
                    continue;
                };

                // Must be OpVariable with UniformConstant storage class
                if interface_inst.class.opcode != Op::Variable {
                    return Err(ValidationError::GraphEntryPointInterfaceNotVariable {
                        instruction_id: inst.result_id.map(to_id),
                        interface_id: to_id(*interface_id),
                    });
                }

                // Check storage class (operand 0 of OpVariable)
                if let Some(Operand::StorageClass(sc)) = interface_inst.operands.first() {
                    if *sc != StorageClass::UniformConstant {
                        return Err(ValidationError::GraphEntryPointInterfaceNotUniformConstant {
                            instruction_id: inst.result_id.map(to_id),
                            interface_id: to_id(*interface_id),
                        });
                    }
                }

                // Check type matches
                let expected_io_type = if i < graph_type_num_inputs(graph_type_inst) as usize {
                    graph_type_input_at(graph_type_inst, i)
                } else {
                    graph_type_output_at(
                        graph_type_inst,
                        i - graph_type_num_inputs(graph_type_inst) as usize,
                    )
                };

                if let (Some(expected_type), Some(interface_ptr_type)) =
                    (expected_io_type, interface_inst.result_type)
                {
                    // Get pointee type from pointer type
                    if let Some(ptr_inst) = ResultId::try_from(interface_ptr_type)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                    {
                        // OpTypePointer has storage class at operand 0 and pointee type at operand 1
                        if let Some(Operand::IdRef(pointee_type)) = ptr_inst.operands.get(1) {
                            if *pointee_type != expected_type {
                                return Err(ValidationError::GraphEntryPointInterfaceTypeMismatch {
                                    instruction_id: inst.result_id.map(to_id),
                                    interface_id: to_id(*interface_id),
                                    expected_type: to_id(expected_type),
                                    actual_type: to_id(*pointee_type),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpGraphInputARM instructions.
///
/// - InputIndex must be a 32-bit integer
/// - ElementIndex (if present) must be a 32-bit integer
/// - InputIndex must be in range for the graph type
/// - ElementIndex only allowed for array inputs
/// - Result type must match the graph input type
pub struct GraphInputRule;

impl ValidationRule for GraphInputRule {
    fn name(&self) -> &'static str {
        "graph-input"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Iterate through functions to find graph inputs within graph definitions
        for func in &ctx.module.functions {
            for block in &func.blocks {
                let mut current_graph_type: Option<&rspirv::dr::Instruction> = None;

                for inst in &block.instructions {
                    // Track current graph definition
                    if inst.class.opcode == Op::GraphARM {
                        if let Some(type_id) = inst.result_type {
                            current_graph_type = ResultId::try_from(type_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));
                        }
                    }

                    if inst.class.opcode == Op::GraphEndARM {
                        current_graph_type = None;
                    }

                    if inst.class.opcode != Op::GraphInputARM {
                        continue;
                    }

                    // Check InputIndex type (operand 0)
                    if let Some(Operand::IdRef(input_index_id)) = inst.operands.first() {
                        if let Some(input_index_inst) = ResultId::try_from(*input_index_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                        {
                            if let Some(type_id) = input_index_inst.result_type {
                                if let Some(type_inst) = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                {
                                    // Must be 32-bit integer
                                    if type_inst.class.opcode != Op::TypeInt {
                                        return Err(ValidationError::GraphInputIndexNotInt32 {
                                            instruction_id: inst.result_id.map(to_id),
                                            operand: "InputIndex",
                                        });
                                    }
                                    // Check width is 32
                                    if let Some(Operand::LiteralBit32(width)) =
                                        type_inst.operands.first()
                                    {
                                        if *width != 32 {
                                            return Err(ValidationError::GraphInputIndexNotInt32 {
                                                instruction_id: inst.result_id.map(to_id),
                                                operand: "InputIndex",
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Check ElementIndex type if present (operand 1)
                    if let Some(Operand::IdRef(element_index_id)) = inst.operands.get(1) {
                        if let Some(element_index_inst) = ResultId::try_from(*element_index_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                        {
                            if let Some(type_id) = element_index_inst.result_type {
                                if let Some(type_inst) = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                {
                                    if type_inst.class.opcode != Op::TypeInt {
                                        return Err(ValidationError::GraphInputIndexNotInt32 {
                                            instruction_id: inst.result_id.map(to_id),
                                            operand: "ElementIndex",
                                        });
                                    }
                                    if let Some(Operand::LiteralBit32(width)) =
                                        type_inst.operands.first()
                                    {
                                        if *width != 32 {
                                            return Err(ValidationError::GraphInputIndexNotInt32 {
                                                instruction_id: inst.result_id.map(to_id),
                                                operand: "ElementIndex",
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Validate against graph type if we know it
                    if let Some(graph_type_inst) = current_graph_type {
                        if let Some(Operand::IdRef(input_index_id)) = inst.operands.first() {
                            if let Some(input_index) =
                                eval_constant_u64(*input_index_id, ctx.definitions)
                            {
                                let num_inputs =
                                    graph_type_num_inputs(graph_type_inst) as u64;

                                // Check index in range
                                if input_index >= num_inputs {
                                    return Err(ValidationError::GraphInputIndexOutOfRange {
                                        instruction_id: inst.result_id.map(to_id),
                                        input_index,
                                        num_inputs,
                                    });
                                }

                                let has_element_index = inst.operands.len() > 1;
                                let input_type =
                                    graph_type_input_at(graph_type_inst, input_index as usize);

                                if has_element_index {
                                    // ElementIndex only allowed for tensor arrays
                                    if let Some(type_id) = input_type {
                                        if !is_tensor_array(type_id, ctx.definitions) {
                                            return Err(
                                                ValidationError::GraphInputElementIndexNotAllowed {
                                                    instruction_id: inst.result_id.map(to_id),
                                                },
                                            );
                                        }
                                    }
                                }

                                // Check result type matches
                                if let (Some(result_type), Some(expected_type)) =
                                    (inst.result_type, input_type)
                                {
                                    let expected = if has_element_index {
                                        // Get element type of array
                                        if let Some(array_inst) = ResultId::try_from(expected_type)
                                            .ok()
                                            .and_then(|rid| ctx.definitions.get(&rid))
                                        {
                                            if let Some(Operand::IdRef(elem_type)) =
                                                array_inst.operands.first()
                                            {
                                                *elem_type
                                            } else {
                                                expected_type
                                            }
                                        } else {
                                            expected_type
                                        }
                                    } else {
                                        expected_type
                                    };

                                    if result_type != expected {
                                        return Err(ValidationError::GraphInputTypeMismatch {
                                            instruction_id: inst.result_id.map(to_id),
                                            expected_type: to_id(expected),
                                            actual_type: to_id(result_type),
                                        });
                                    }
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

/// Validates OpGraphSetOutputARM instructions.
///
/// - OutputIndex must be a 32-bit integer
/// - ElementIndex (if present) must be a 32-bit integer
/// - OutputIndex must be in range for the graph type
/// - ElementIndex only allowed for array outputs
/// - Value type must match the graph output type
pub struct GraphSetOutputRule;

impl ValidationRule for GraphSetOutputRule {
    fn name(&self) -> &'static str {
        "graph-set-output"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            for block in &func.blocks {
                let mut current_graph_type: Option<&rspirv::dr::Instruction> = None;

                for inst in &block.instructions {
                    if inst.class.opcode == Op::GraphARM {
                        if let Some(type_id) = inst.result_type {
                            current_graph_type = ResultId::try_from(type_id)
                                .ok()
                                .and_then(|rid| ctx.definitions.get(&rid));
                        }
                    }

                    if inst.class.opcode == Op::GraphEndARM {
                        current_graph_type = None;
                    }

                    if inst.class.opcode != Op::GraphSetOutputARM {
                        continue;
                    }

                    // OpGraphSetOutputARM: Value, OutputIndex, [ElementIndex]
                    // Check OutputIndex type (operand 1)
                    if let Some(Operand::IdRef(output_index_id)) = inst.operands.get(1) {
                        if let Some(output_index_inst) = ResultId::try_from(*output_index_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                        {
                            if let Some(type_id) = output_index_inst.result_type {
                                if let Some(type_inst) = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                {
                                    if type_inst.class.opcode != Op::TypeInt {
                                        return Err(ValidationError::GraphOutputIndexNotInt32 {
                                            instruction_id: inst.result_id.map(to_id),
                                            operand: "OutputIndex",
                                        });
                                    }
                                    if let Some(Operand::LiteralBit32(width)) =
                                        type_inst.operands.first()
                                    {
                                        if *width != 32 {
                                            return Err(ValidationError::GraphOutputIndexNotInt32 {
                                                instruction_id: inst.result_id.map(to_id),
                                                operand: "OutputIndex",
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Check ElementIndex type if present (operand 2)
                    if let Some(Operand::IdRef(element_index_id)) = inst.operands.get(2) {
                        if let Some(element_index_inst) = ResultId::try_from(*element_index_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                        {
                            if let Some(type_id) = element_index_inst.result_type {
                                if let Some(type_inst) = ResultId::try_from(type_id)
                                    .ok()
                                    .and_then(|rid| ctx.definitions.get(&rid))
                                {
                                    if type_inst.class.opcode != Op::TypeInt {
                                        return Err(ValidationError::GraphOutputIndexNotInt32 {
                                            instruction_id: inst.result_id.map(to_id),
                                            operand: "ElementIndex",
                                        });
                                    }
                                    if let Some(Operand::LiteralBit32(width)) =
                                        type_inst.operands.first()
                                    {
                                        if *width != 32 {
                                            return Err(ValidationError::GraphOutputIndexNotInt32 {
                                                instruction_id: inst.result_id.map(to_id),
                                                operand: "ElementIndex",
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Validate against graph type if we know it
                    if let Some(graph_type_inst) = current_graph_type {
                        if let Some(Operand::IdRef(output_index_id)) = inst.operands.get(1) {
                            if let Some(output_index) =
                                eval_constant_u64(*output_index_id, ctx.definitions)
                            {
                                let num_outputs = graph_type_num_outputs(graph_type_inst) as u64;

                                if output_index >= num_outputs {
                                    return Err(ValidationError::GraphOutputIndexOutOfRange {
                                        instruction_id: inst.result_id.map(to_id),
                                        output_index,
                                        num_outputs,
                                    });
                                }

                                let has_element_index = inst.operands.len() > 2;
                                let output_type =
                                    graph_type_output_at(graph_type_inst, output_index as usize);

                                if has_element_index {
                                    if let Some(type_id) = output_type {
                                        if !is_tensor_array(type_id, ctx.definitions) {
                                            return Err(
                                                ValidationError::GraphOutputElementIndexNotAllowed {
                                                    instruction_id: inst.result_id.map(to_id),
                                                },
                                            );
                                        }
                                    }
                                }

                                // Check Value type matches
                                if let Some(Operand::IdRef(value_id)) = inst.operands.first() {
                                    if let Some(value_inst) = ResultId::try_from(*value_id)
                                        .ok()
                                        .and_then(|rid| ctx.definitions.get(&rid))
                                    {
                                        if let (Some(value_type), Some(expected_type)) =
                                            (value_inst.result_type, output_type)
                                        {
                                            let expected = if has_element_index {
                                                if let Some(array_inst) =
                                                    ResultId::try_from(expected_type)
                                                        .ok()
                                                        .and_then(|rid| ctx.definitions.get(&rid))
                                                {
                                                    if let Some(Operand::IdRef(elem_type)) =
                                                        array_inst.operands.first()
                                                    {
                                                        *elem_type
                                                    } else {
                                                        expected_type
                                                    }
                                                } else {
                                                    expected_type
                                                }
                                            } else {
                                                expected_type
                                            };

                                            if value_type != expected {
                                                return Err(
                                                    ValidationError::GraphOutputTypeMismatch {
                                                        instruction_id: inst.result_id.map(to_id),
                                                        expected_type: to_id(expected),
                                                        actual_type: to_id(value_type),
                                                    },
                                                );
                                            }
                                        }
                                    }
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

/// Validates OpGraphEndARM instructions.
///
/// - No duplicate InputIndex values within the same graph definition
/// - No duplicate OutputIndex values within the same graph definition
pub struct GraphEndRule;

impl ValidationRule for GraphEndRule {
    fn name(&self) -> &'static str {
        "graph-end"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            for block in &func.blocks {
                let mut in_graph = false;
                let mut input_indices: HashSet<(u64, Option<u64>)> = HashSet::new();
                let mut output_indices: HashSet<(u64, Option<u64>)> = HashSet::new();

                for inst in &block.instructions {
                    if inst.class.opcode == Op::GraphARM {
                        in_graph = true;
                        input_indices.clear();
                        output_indices.clear();
                    }

                    if !in_graph {
                        continue;
                    }

                    if inst.class.opcode == Op::GraphInputARM {
                        if let Some(Operand::IdRef(input_index_id)) = inst.operands.first() {
                            if let Some(input_index) =
                                eval_constant_u64(*input_index_id, ctx.definitions)
                            {
                                let element_index = inst.operands.get(1).and_then(|op| {
                                    if let Operand::IdRef(id) = op {
                                        eval_constant_u64(*id, ctx.definitions)
                                    } else {
                                        None
                                    }
                                });

                                let key = (input_index, element_index);
                                let no_element_key = (input_index, None);

                                // Duplicate if same key exists or key with no element exists
                                if input_indices.contains(&key)
                                    || (element_index.is_some()
                                        && input_indices.contains(&no_element_key))
                                    || (element_index.is_none()
                                        && input_indices
                                            .iter()
                                            .any(|(idx, _)| *idx == input_index))
                                {
                                    return Err(ValidationError::GraphDuplicateInputIndex {
                                        instruction_id: inst.result_id.map(to_id),
                                        input_index,
                                    });
                                }
                                input_indices.insert(key);
                            }
                        }
                    }

                    if inst.class.opcode == Op::GraphSetOutputARM {
                        if let Some(Operand::IdRef(output_index_id)) = inst.operands.get(1) {
                            if let Some(output_index) =
                                eval_constant_u64(*output_index_id, ctx.definitions)
                            {
                                let element_index = inst.operands.get(2).and_then(|op| {
                                    if let Operand::IdRef(id) = op {
                                        eval_constant_u64(*id, ctx.definitions)
                                    } else {
                                        None
                                    }
                                });

                                let key = (output_index, element_index);
                                let no_element_key = (output_index, None);

                                if output_indices.contains(&key)
                                    || (element_index.is_some()
                                        && output_indices.contains(&no_element_key))
                                    || (element_index.is_none()
                                        && output_indices
                                            .iter()
                                            .any(|(idx, _)| *idx == output_index))
                                {
                                    return Err(ValidationError::GraphDuplicateOutputIndex {
                                        instruction_id: inst.result_id.map(to_id),
                                        output_index,
                                    });
                                }
                                output_indices.insert(key);
                            }
                        }
                    }

                    if inst.class.opcode == Op::GraphEndARM {
                        in_graph = false;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all ARM graph validation rules.
pub fn all_graph_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(GraphTypeRule),
        Box::new(GraphConstantRule),
        Box::new(GraphRule),
        Box::new(GraphEntryPointRule),
        Box::new(GraphInputRule),
        Box::new(GraphSetOutputRule),
        Box::new(GraphEndRule),
    ]
}
