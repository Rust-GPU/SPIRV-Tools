//! Shared helper functions for SPIR-V validation.
//!
//! This module provides common utility functions used across multiple
//! validation rules, including type inspection, decoration lookup,
//! and instruction parsing helpers.

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Capability, Decoration, ExecutionModel, Op, StorageClass};

use super::types::{IdKind, ResultId, TypeId};
use super::ValidationError;
use crate::target_env::TargetEnv;

// ============================================================================
// Type inspection helpers
// ============================================================================

/// Returns true if the opcode defines a type.
pub fn is_type_opcode(opcode: Op) -> bool {
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
            | Op::TypeFunction
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
            | Op::TypePipeStorage
            | Op::TypeNamedBarrier
            | Op::TypeRayQueryKHR
            | Op::TypeAccelerationStructureKHR
            | Op::TypeCooperativeMatrixNV
            | Op::TypeCooperativeMatrixKHR
            | Op::TypeHitObjectNV
            | Op::TypeUntypedPointerKHR
    )
}

/// Returns true if the instruction is a void type.
pub fn is_void_type(
    type_id: TypeId,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    let result_id = ResultId::try_from(u32::from(type_id)).ok();
    result_id
        .and_then(|id| definitions.get(&id))
        .is_some_and(|inst| inst.class.opcode == Op::TypeVoid)
}

/// Returns true if the instruction is a boolean type.
pub fn is_bool(inst: &Instruction) -> bool {
    inst.class.opcode == Op::TypeBool
}

/// Returns true if the instruction is a 32-bit float.
pub fn is_float32(inst: &Instruction) -> bool {
    inst.class.opcode == Op::TypeFloat
        && inst.operands.first() == Some(&Operand::LiteralBit32(32))
}

/// Returns true if the instruction is a 32-bit integer.
pub fn is_int32(inst: &Instruction) -> bool {
    inst.class.opcode == Op::TypeInt
        && inst.operands.first() == Some(&Operand::LiteralBit32(32))
}

/// Returns the bit width of a numeric type.
pub fn type_bit_width(ty: &Instruction) -> Option<u32> {
    match ty.class.opcode {
        Op::TypeInt | Op::TypeFloat => literal_u32(ty.operands.first()?),
        _ => None,
    }
}

/// Returns true if the instruction is a float scalar of the specified width.
pub fn is_float_scalar_of_width(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    width: u32,
) -> bool {
    if inst.class.opcode == Op::TypeFloat {
        return inst.operands.first() == Some(&Operand::LiteralBit32(width));
    }
    if inst.class.opcode == Op::TypeVector {
        if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
            if let Ok(elem_result) = ResultId::try_from(*elem_id) {
                if let Some(elem_inst) = definitions.get(&elem_result) {
                    return is_float_scalar_of_width(elem_inst, definitions, width);
                }
            }
        }
    }
    false
}

/// Returns true if the instruction is an integer scalar or vector.
pub fn is_int_scalar_or_vector(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    match inst.class.opcode {
        Op::TypeInt => true,
        Op::TypeVector => {
            if let Some(Operand::IdRef(elem_id)) = inst.operands.first() {
                if let Ok(elem_result) = ResultId::try_from(*elem_id) {
                    if let Some(elem_inst) = definitions.get(&elem_result) {
                        return elem_inst.class.opcode == Op::TypeInt;
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Returns the bit width of an integer type (scalar or vector).
pub fn int_bit_width(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    match inst.class.opcode {
        Op::TypeInt => literal_u32(inst.operands.first()?),
        Op::TypeVector => {
            let Operand::IdRef(elem_id) = inst.operands.first()? else {
                return None;
            };
            let elem_result = ResultId::try_from(*elem_id).ok()?;
            let elem_inst = definitions.get(&elem_result)?;
            if elem_inst.class.opcode == Op::TypeInt {
                literal_u32(elem_inst.operands.first()?)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Returns true if the instruction is a vector of the specified element type.
pub fn is_vector_of(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    element_pred: impl Fn(&Instruction) -> bool,
) -> bool {
    if inst.class.opcode != Op::TypeVector {
        return false;
    }
    let Some(Operand::IdRef(elem_id)) = inst.operands.first() else {
        return false;
    };
    let Ok(elem_result) = ResultId::try_from(*elem_id) else {
        return false;
    };
    let Some(elem_inst) = definitions.get(&elem_result) else {
        return false;
    };
    element_pred(elem_inst)
}

/// Returns true if the instruction is an array of the specified element type.
pub fn is_array_of(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
    element_pred: impl Fn(&Instruction) -> bool,
) -> bool {
    if inst.class.opcode != Op::TypeArray {
        return false;
    }
    let Some(Operand::IdRef(elem_id)) = inst.operands.first() else {
        return false;
    };
    let Ok(elem_result) = ResultId::try_from(*elem_id) else {
        return false;
    };
    let Some(elem_inst) = definitions.get(&elem_result) else {
        return false;
    };
    element_pred(elem_inst)
}

/// Returns (component_type_id, component_count) for a vector instruction.
///
/// Note: This does not check the opcode - caller should verify this is a TypeVector.
pub fn vector_info(inst: &Instruction) -> (Option<TypeId>, Option<u32>) {
    let component_type = inst
        .operands
        .first()
        .and_then(|op| match op {
            Operand::IdRef(id) => TypeId::try_from(*id).ok(),
            _ => None,
        });
    let count = inst.operands.get(1).and_then(|op| match op {
        Operand::LiteralBit32(v) => Some(*v),
        Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
        _ => None,
    });
    (component_type, count)
}

/// Returns (column_type_id, column_count) for a matrix instruction.
///
/// Note: This does not check the opcode - caller should verify this is a TypeMatrix.
pub fn matrix_info(inst: &Instruction) -> (Option<TypeId>, Option<u32>) {
    let column_type = inst
        .operands
        .first()
        .and_then(|op| match op {
            Operand::IdRef(id) => TypeId::try_from(*id).ok(),
            _ => None,
        });
    let count = inst.operands.get(1).and_then(|op| match op {
        Operand::LiteralBit32(v) => Some(*v),
        Operand::LiteralBit64(v) => u32::try_from(*v).ok(),
        _ => None,
    });
    (column_type, count)
}

/// Returns detailed matrix info: (column_type, column_count, component_type, component_count).
pub fn matrix_details(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> (Option<TypeId>, Option<u32>, Option<TypeId>, Option<u32>) {
    let (column_type, column_count) = matrix_info(inst);
    let (component_type, component_count) = column_type
        .and_then(|ty| {
            let result_id = ResultId::try_from(u32::from(ty)).ok()?;
            definitions.get(&result_id)
        })
        .map(vector_info)
        .unwrap_or((None, None));
    (column_type, column_count, component_type, component_count)
}

/// Returns matrix details by type id: (component_type, rows, columns, column_result_id).
/// This is the original validation mod.rs signature used for matrix validation.
pub fn matrix_details_by_id(
    type_id: TypeId,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<(TypeId, u32, u32, ResultId)> {
    let matrix_result = ResultId::try_from(u32::from(type_id)).ok()?;
    let inst = definitions.get(&matrix_result)?;
    if inst.class.opcode != Op::TypeMatrix {
        return None;
    }
    let (column_type, columns) = matrix_info(inst);
    let column_type = column_type?;
    let columns = columns?;
    let column_result = ResultId::try_from(u32::from(column_type)).ok()?;
    let column_inst = definitions.get(&column_result)?;
    if column_inst.class.opcode != Op::TypeVector {
        return None;
    }
    let (component_type, rows) = vector_info(column_inst);
    Some((component_type?, rows?, columns, column_result))
}

/// Returns the length of a fixed-size array from its length operand.
pub fn array_length(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    let len_id = match inst.operands.get(1) {
        Some(Operand::IdRef(id)) => ResultId::try_from(*id).ok()?,
        _ => return None,
    };
    let len_inst = definitions.get(&len_id)?;
    if len_inst.class.opcode != Op::Constant {
        return None;
    }
    match len_inst.operands.first() {
        Some(Operand::LiteralBit32(v)) => Some(*v),
        Some(Operand::LiteralBit64(v)) => u32::try_from(*v).ok(),
        _ => None,
    }
}

// ============================================================================
// Operand extraction helpers
// ============================================================================

/// Extracts a u32 literal from an operand.
pub fn literal_u32(op: &Operand) -> Option<u32> {
    match op {
        Operand::LiteralBit32(v) => Some(*v),
        _ => None,
    }
}

/// Extracts an IdRef from an operand.
pub fn id_ref(op: &Operand) -> Option<u32> {
    match op {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

// ============================================================================
// Decoration helpers
// ============================================================================

/// Returns true if the target has the specified decoration.
pub fn has_decoration(module: &Module, target: u32, decoration: Decoration) -> bool {
    module.annotations.iter().any(|inst| {
        if inst.class.opcode == Op::Decorate {
            if let (Some(Operand::IdRef(id)), Some(Operand::Decoration(dec))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                return *id == target && *dec == decoration;
            }
        }
        false
    })
}

/// Returns true if the target has a Block decoration.
pub fn has_block_decoration(module: &Module, type_id: ResultId) -> bool {
    has_decoration(module, u32::from(type_id), Decoration::Block)
}

/// Returns true if the target has a Patch decoration.
pub fn has_patch_decoration(module: &Module, target: ResultId) -> bool {
    has_decoration(module, u32::from(target), Decoration::Patch)
}

/// Returns the Location and Component decorations for a target.
pub fn location_and_component(module: &Module, target: ResultId) -> Option<(u32, u32)> {
    let id = u32::from(target);
    let mut location = None;
    let mut component = 0u32;
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let Some(Operand::IdRef(target_id)) = inst.operands.first() {
                if *target_id == id {
                    if let Some(Operand::Decoration(dec)) = inst.operands.get(1) {
                        match dec {
                            Decoration::Location => {
                                location = inst.operands.get(2).and_then(literal_u32);
                            }
                            Decoration::Component => {
                                component = inst.operands.get(2).and_then(literal_u32).unwrap_or(0);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    location.map(|loc| (loc, component))
}

/// Builds a lookup table of decorations for each result ID.
pub fn build_decoration_lookup(module: &Module) -> HashMap<ResultId, Vec<Decoration>> {
    let mut result: HashMap<ResultId, Vec<Decoration>> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate {
            if let (Some(Operand::IdRef(id)), Some(Operand::Decoration(dec))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if let Ok(result_id) = ResultId::try_from(*id) {
                    result.entry(result_id).or_default().push(*dec);
                }
            }
        }
    }
    result
}

// ============================================================================
// Pointer/storage class helpers
// ============================================================================

/// Returns true if the instruction defines a pointer type.
pub fn is_pointer_type(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> bool {
    match inst.class.opcode {
        Op::TypePointer | Op::TypeUntypedPointerKHR => true,
        Op::Variable | Op::UntypedVariableKHR => {
            if let Some(type_id) = inst.result_type {
                if let Ok(result_id) = ResultId::try_from(type_id) {
                    if let Some(type_inst) = definitions.get(&result_id) {
                        return is_pointer_type(type_inst, definitions);
                    }
                }
            }
            false
        }
        _ => false,
    }
}

/// Returns (storage_class, pointee_type_id) for a pointer type.
pub fn pointer_info(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> (Option<StorageClass>, Option<TypeId>) {
    match inst.class.opcode {
        Op::TypePointer => {
            let storage = inst.operands.first().and_then(|op| match op {
                Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            });
            let pointee = inst.operands.get(1).and_then(|op| match op {
                Operand::IdRef(id) => TypeId::try_from(*id).ok(),
                _ => None,
            });
            (storage, pointee)
        }
        Op::TypeUntypedPointerKHR => {
            let storage = inst.operands.first().and_then(|op| match op {
                Operand::StorageClass(sc) => Some(*sc),
                _ => None,
            });
            (storage, None)
        }
        Op::Variable | Op::UntypedVariableKHR => {
            if let Some(type_id) = inst.result_type {
                if let Ok(result_id) = ResultId::try_from(type_id) {
                    if let Some(type_inst) = definitions.get(&result_id) {
                        return pointer_info(type_inst, definitions);
                    }
                }
            }
            (None, None)
        }
        _ => (None, None),
    }
}

// ============================================================================
// Instruction classification helpers
// ============================================================================

/// Returns true if the opcode is a scalar specialization constant.
pub fn is_scalar_spec_constant(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::SpecConstantTrue | Op::SpecConstantFalse | Op::SpecConstant
    )
}

/// Returns true if the opcode defines any kind of constant.
pub fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::ConstantTrue
            | Op::ConstantFalse
            | Op::Constant
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::ConstantNull
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstant
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
    )
}

/// Returns true if the opcode is a memory object declaration.
pub fn is_memory_object_declaration(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Variable
            | Op::UntypedVariableKHR
            | Op::FunctionParameter
            | Op::RawAccessChainNV
    )
}

// ============================================================================
// Module collection helpers
// ============================================================================

/// Collects all execution models from entry points.
pub fn collect_execution_models(module: &Module) -> HashSet<ExecutionModel> {
    module
        .entry_points
        .iter()
        .filter_map(|inst| {
            inst.operands.first().and_then(|op| match op {
                Operand::ExecutionModel(em) => Some(*em),
                _ => None,
            })
        })
        .collect()
}

/// Collects result IDs mapped to their opcodes.
pub fn collect_result_opcodes(module: &Module) -> HashMap<ResultId, Op> {
    let mut result = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            if let Ok(result_id) = ResultId::try_from(id) {
                result.insert(result_id, inst.class.opcode);
            }
        }
    }
    result
}

/// Collects result IDs mapped to their full instructions.
pub fn collect_result_instructions(module: &Module) -> HashMap<ResultId, Instruction> {
    let mut result = HashMap::new();
    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            if let Ok(result_id) = ResultId::try_from(id) {
                result.insert(result_id, inst.clone());
            }
        }
    }
    result
}

/// Collects result IDs mapped to their result type IDs.
pub fn collect_result_types(
    module: &Module,
) -> Result<HashMap<ResultId, TypeId>, ValidationError> {
    let mut result = HashMap::new();
    for inst in module.all_inst_iter() {
        if let (Some(result_id), Some(type_id)) = (inst.result_id, inst.result_type) {
            let result_id = ResultId::try_from(result_id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::Result,
                opcode: inst.class.opcode,
            })?;
            let type_id = TypeId::try_from(type_id).map_err(|_| ValidationError::ZeroId {
                kind: IdKind::ResultType,
                opcode: inst.class.opcode,
            })?;
            result.insert(result_id, type_id);
        }
    }
    Ok(result)
}

/// Collects all declared capabilities.
pub fn collect_declared_capabilities(module: &Module) -> HashSet<Capability> {
    module
        .capabilities
        .iter()
        .filter_map(|inst| {
            inst.operands.first().and_then(|op| match op {
                Operand::Capability(cap) => Some(*cap),
                _ => None,
            })
        })
        .collect()
}

// ============================================================================
// Environment helpers
// ============================================================================

/// Returns true if the environment is a Vulkan variant.
pub fn is_vulkan_env(env: TargetEnv) -> bool {
    matches!(
        env,
        TargetEnv::Vulkan1_0
            | TargetEnv::Vulkan1_1
            | TargetEnv::Vulkan1_1Spirv1_4
            | TargetEnv::Vulkan1_2
            | TargetEnv::Vulkan1_3
            | TargetEnv::Vulkan1_4
    )
}

// ============================================================================
// Constant evaluation helpers
// ============================================================================

/// Returns a u32 constant value from the module by ID.
pub fn constant_u32(module: &Module, id: u32) -> Option<u32> {
    module.types_global_values.iter().find_map(|inst| {
        if inst.result_id == Some(id) && inst.class.opcode == Op::Constant {
            literal_u32(inst.operands.first()?)
        } else {
            None
        }
    })
}

/// Returns a u32 constant value from definitions by ResultId.
pub fn constant_u32_from_defs(
    definitions: &HashMap<ResultId, Instruction>,
    id: ResultId,
) -> Option<u32> {
    let inst = definitions.get(&id)?;
    if inst.class.opcode == Op::Constant {
        literal_u32(inst.operands.first()?)
    } else {
        None
    }
}

// ============================================================================
// Type structure parsing
// ============================================================================

use super::types::{BitWidth, MatrixColumns, ScalarKind, TypeStructure, VectorSize};

/// Parses an instruction defining a type into a `TypeStructure`.
pub fn parse_type_structure(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> TypeStructure {
    match inst.class.opcode {
        Op::TypeVoid => TypeStructure::Void,
        Op::TypeBool => TypeStructure::Scalar(ScalarKind::Bool),
        Op::TypeInt => parse_int_type(inst),
        Op::TypeFloat => parse_float_type(inst),
        Op::TypeVector => parse_vector_type(inst, definitions),
        Op::TypeMatrix => parse_matrix_type(inst, definitions),
        Op::TypeArray => parse_array_type(inst),
        Op::TypeRuntimeArray => parse_runtime_array_type(inst),
        Op::TypeStruct => parse_struct_type(inst),
        Op::TypePointer => parse_pointer_type(inst),
        Op::TypeUntypedPointerKHR => parse_untyped_pointer_type(inst),
        Op::TypeImage => parse_image_type(inst),
        Op::TypeSampler => TypeStructure::Sampler,
        Op::TypeSampledImage => parse_sampled_image_type(inst),
        Op::TypeFunction => parse_function_type(inst),
        Op::TypeCooperativeMatrixKHR | Op::TypeCooperativeMatrixNV => {
            parse_cooperative_matrix_type(inst)
        }
        Op::TypeCooperativeVectorNV => parse_cooperative_vector_type(inst),
        Op::TypeForwardPointer => parse_forward_pointer_type(inst),
        Op::TypeOpaque => TypeStructure::Opaque,
        _ => TypeStructure::Unknown,
    }
}

fn parse_int_type(inst: &Instruction) -> TypeStructure {
    let width = inst
        .operands
        .first()
        .and_then(literal_u32)
        .and_then(BitWidth::new);
    let signedness = inst.operands.get(1).and_then(literal_u32).unwrap_or(1);

    match (width, signedness) {
        (Some(w), 0) => TypeStructure::Scalar(ScalarKind::UnsignedInt(w)),
        (Some(w), _) => TypeStructure::Scalar(ScalarKind::SignedInt(w)),
        _ => TypeStructure::Unknown,
    }
}

fn parse_float_type(inst: &Instruction) -> TypeStructure {
    inst.operands
        .first()
        .and_then(literal_u32)
        .and_then(BitWidth::new)
        .map(|w| TypeStructure::Scalar(ScalarKind::Float(w)))
        .unwrap_or(TypeStructure::Unknown)
}

fn parse_vector_type(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> TypeStructure {
    let component_id = inst.operands.first().and_then(id_ref);
    let size = inst.operands.get(1).and_then(literal_u32).and_then(VectorSize::new);

    let component = component_id
        .and_then(|id| ResultId::try_from(id).ok())
        .and_then(|id| definitions.get(&id))
        .map(|comp_inst| parse_type_structure(comp_inst, definitions))
        .and_then(|ts| match ts {
            TypeStructure::Scalar(k) => Some(k),
            _ => None,
        });

    match (component, size) {
        (Some(c), Some(s)) => TypeStructure::Vector { component: c, size: s },
        _ => TypeStructure::Unknown,
    }
}

fn parse_matrix_type(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> TypeStructure {
    let column_id = inst.operands.first().and_then(id_ref);
    let cols = inst.operands.get(1).and_then(literal_u32).and_then(MatrixColumns::new);

    let (component, rows) = column_id
        .and_then(|id| ResultId::try_from(id).ok())
        .and_then(|id| definitions.get(&id))
        .map(|col_inst| parse_type_structure(col_inst, definitions))
        .and_then(|ts| match ts {
            TypeStructure::Vector { component, size } => Some((component, size)),
            _ => None,
        })
        .unzip();

    match (component, rows, cols) {
        (Some(c), Some(r), Some(col)) => TypeStructure::Matrix {
            component: c,
            rows: r,
            cols: col,
        },
        _ => TypeStructure::Unknown,
    }
}

fn parse_array_type(inst: &Instruction) -> TypeStructure {
    let element = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match element {
        Some(e) => TypeStructure::Array {
            element: e,
            length: None, // Length evaluation would need constant folding
        },
        None => TypeStructure::Unknown,
    }
}

fn parse_runtime_array_type(inst: &Instruction) -> TypeStructure {
    let element = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match element {
        Some(e) => TypeStructure::RuntimeArray { element: e },
        None => TypeStructure::Unknown,
    }
}

fn parse_struct_type(inst: &Instruction) -> TypeStructure {
    let members: Vec<TypeId> = inst
        .operands
        .iter()
        .filter_map(id_ref)
        .filter_map(|id| TypeId::try_from(id).ok())
        .collect();

    TypeStructure::Struct { members }
}

fn parse_pointer_type(inst: &Instruction) -> TypeStructure {
    let storage_class = inst.operands.first().and_then(|op| match op {
        Operand::StorageClass(sc) => Some(*sc),
        _ => None,
    });
    let pointee = inst
        .operands
        .get(1)
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match storage_class {
        Some(sc) => TypeStructure::Pointer {
            pointee,
            storage_class: sc,
        },
        None => TypeStructure::Unknown,
    }
}

fn parse_untyped_pointer_type(inst: &Instruction) -> TypeStructure {
    let storage_class = inst.operands.first().and_then(|op| match op {
        Operand::StorageClass(sc) => Some(*sc),
        _ => None,
    });

    match storage_class {
        Some(sc) => TypeStructure::Pointer {
            pointee: None,
            storage_class: sc,
        },
        None => TypeStructure::Unknown,
    }
}

fn parse_image_type(inst: &Instruction) -> TypeStructure {
    let sampled_type = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    TypeStructure::Image { sampled_type }
}

fn parse_sampled_image_type(inst: &Instruction) -> TypeStructure {
    let image_type = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match image_type {
        Some(it) => TypeStructure::SampledImage { image_type: it },
        None => TypeStructure::Unknown,
    }
}

fn parse_function_type(inst: &Instruction) -> TypeStructure {
    let return_type = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    let params: Vec<TypeId> = inst
        .operands
        .iter()
        .skip(1)
        .filter_map(id_ref)
        .filter_map(|id| TypeId::try_from(id).ok())
        .collect();

    match return_type {
        Some(rt) => TypeStructure::Function {
            return_type: rt,
            params,
        },
        None => TypeStructure::Unknown,
    }
}

fn parse_cooperative_matrix_type(inst: &Instruction) -> TypeStructure {
    let component = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match component {
        Some(c) => TypeStructure::CooperativeMatrix { component: c },
        None => TypeStructure::Unknown,
    }
}

fn parse_cooperative_vector_type(inst: &Instruction) -> TypeStructure {
    let component = inst
        .operands
        .first()
        .and_then(id_ref)
        .and_then(|id| TypeId::try_from(id).ok());

    match component {
        Some(c) => TypeStructure::CooperativeVector { component: c },
        None => TypeStructure::Unknown,
    }
}

fn parse_forward_pointer_type(inst: &Instruction) -> TypeStructure {
    let storage_class = inst.operands.get(1).and_then(|op| match op {
        Operand::StorageClass(sc) => Some(*sc),
        _ => None,
    });

    match storage_class {
        Some(sc) => TypeStructure::ForwardPointer { storage_class: sc },
        None => TypeStructure::Unknown,
    }
}

/// Gets the type structure for a given type ID.
pub fn get_type_structure(
    type_id: TypeId,
    definitions: &HashMap<ResultId, Instruction>,
) -> TypeStructure {
    ResultId::try_from(u32::from(type_id))
        .ok()
        .and_then(|rid| definitions.get(&rid))
        .map(|inst| parse_type_structure(inst, definitions))
        .unwrap_or(TypeStructure::Unknown)
}

/// Gets the type structure for the result type of an instruction.
pub fn get_result_type_structure(
    inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<TypeStructure> {
    inst.result_type
        .and_then(|raw| TypeId::try_from(raw).ok())
        .map(|tid| get_type_structure(tid, definitions))
}

/// Gets the type structure for an operand at a given index.
pub fn get_operand_type_structure(
    inst: &Instruction,
    operand_index: usize,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<TypeStructure> {
    let operand_id = inst.operands.get(operand_index).and_then(id_ref)?;
    let operand_result_id = ResultId::try_from(operand_id).ok()?;
    let operand_inst = definitions.get(&operand_result_id)?;
    let operand_type_id = TypeId::try_from(operand_inst.result_type?).ok()?;
    Some(get_type_structure(operand_type_id, definitions))
}

/// Calculates how many location components a type consumes.
///
/// This is used for interface location validation. Different types consume
/// different numbers of location slots:
/// - Scalars (int, float): 1 component
/// - Vectors: component_count components
/// - Matrices: column_count * components_per_column
/// - Arrays: element_count * components_per_element
/// - Structs: sum of all member components
///
/// Returns None for types with runtime-sized arrays or other indeterminate sizes.
pub fn consumed_components_for_type(
    ty: ResultId,
    definitions: &HashMap<ResultId, Instruction>,
    seen: &mut HashSet<ResultId>,
) -> Option<u32> {
    // Prevent infinite recursion
    if !seen.insert(ty) {
        return Some(0);
    }

    let inst = definitions.get(&ty)?;
    match inst.class.opcode {
        Op::TypeInt | Op::TypeFloat => Some(1),
        Op::TypeVector => inst.operands.get(1).and_then(|op| match op {
            Operand::LiteralBit32(count) => Some(*count),
            _ => None,
        }),
        Op::TypeMatrix => {
            let column_type = inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let columns = inst.operands.get(1).and_then(|op| match op {
                Operand::LiteralBit32(count) => Some(*count),
                _ => None,
            })?;
            let mut seen = seen.clone();
            consumed_components_for_type(column_type, definitions, &mut seen)
                .map(|per_column| per_column.saturating_mul(columns))
        }
        Op::TypeArray => {
            let element = inst.operands.first().and_then(|op| match op {
                Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let length_id = inst.operands.get(1).and_then(|op| match op {
                Operand::IdRef(id) => ResultId::try_from(*id).ok(),
                _ => None,
            })?;
            let length = constant_u32_from_defs(definitions, length_id)?;
            let mut seen = seen.clone();
            consumed_components_for_type(element, definitions, &mut seen)
                .map(|per_element| per_element.saturating_mul(length))
        }
        Op::TypeRuntimeArray => None, // Size not known at compile time
        Op::TypeStruct => {
            let mut total: u32 = 0;
            for op in &inst.operands {
                if let Operand::IdRef(member) = op {
                    if let Ok(member_id) = ResultId::try_from(*member) {
                        let mut seen = seen.clone();
                        if let Some(components) =
                            consumed_components_for_type(member_id, definitions, &mut seen)
                        {
                            total = total.saturating_add(components);
                        }
                    }
                }
            }
            Some(total)
        }
        _ => Some(1), // Default for other types
    }
}
