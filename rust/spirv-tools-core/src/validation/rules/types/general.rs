//! General type validation rules.
//!
//! This module validates general SPIR-V type requirements:
//! - Result types must be type opcodes
//! - OpTypeFunction parameter validation
//! - Operand definitions
//! - Type uniqueness
//! - Reserved opcodes
//! - ID pass validation

use std::collections::HashSet;

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::span::ValidationErrorExt;
use crate::validation::types::{Id, ResultId, TypeId};
use crate::validation::ValidationResult;

use super::helpers::is_type_opcode;

// ============================================================================
// Result Types Are Types Rule
// ============================================================================

/// Validates that result_type fields reference actual type instructions.
pub struct ResultTypesAreTypesRule;

impl ValidationRule for ResultTypesAreTypesRule {
    fn name(&self) -> &'static str {
        "result-types-are-types"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in ctx.definitions.values() {
            if let Some(result_type_raw) = inst.result_type {
                if let Ok(type_id) = ResultId::try_from(result_type_raw) {
                    if let Some(type_opcode) = ctx.opcodes.get(&type_id) {
                        if !is_type_opcode(*type_opcode) {
                            let inst_id = inst.result_id.unwrap_or(0);
                            return Err(ValidationError::ResultTypeNotType {
                                instruction: inst.class.opcode,
                                result_type: Id::from(type_id),
                                found: *type_opcode,
                            }
                            .at_ids(
                                inst_id,
                                format!(
                                    "instruction uses non-type {:?} as result type",
                                    type_opcode
                                ),
                                type_id,
                                format!("defined as {:?}, not a type opcode", type_opcode),
                                ctx,
                            ));
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                    return Err(ValidationError::InvalidTypeFunction { type_id }.into());
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
                return Err(ValidationError::InvalidTypeFunction { type_id }.into());
            }

            for op in operands {
                let param_type = match op {
                    rspirv::dr::Operand::IdRef(raw) => TypeId::try_from(*raw)
                        .map_err(|_| ValidationError::InvalidTypeFunction { type_id })?,
                    _ => {
                        return Err(ValidationError::InvalidTypeFunction { type_id }.into());
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
                    }
                    .into());
                }
                if !is_type_opcode(param_opcode) {
                    return Err(ValidationError::InvalidTypeFunction { type_id }.into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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

#[allow(clippy::manual_is_multiple_of)]
fn is_block_operand(opcode: Op, index: usize) -> bool {
    match opcode {
        Op::Branch => index == 0,
        Op::BranchConditional => index == 1 || index == 2,
        Op::Switch => index % 2 == 1,
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
) -> ValidationResult {
    if let Some(result_type) = inst.result_type {
        if let Ok(id) = Id::try_from(result_type) {
            if !defined_ids.contains(&id) {
                return Err(ValidationError::UndefinedId { function, id }.into());
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
                    return Err(ValidationError::UndefinedId { function, id }.into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                            }
                            .into());
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                }
                .into());
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
fn is_non_semantic_instruction(
    inst: &rspirv::dr::Instruction,
    ctx: &ValidationContext<'_>,
) -> bool {
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                operand: Id::try_from(*operand_id)
                                    .unwrap_or(Id::try_from(1u32).unwrap()),
                            }
                            .into());
                        }

                        // Check: Operand must have a type if required
                        if def_inst.result_type.is_none()
                            && !is_type_opcode(def_opcode)
                            && requires_typed_ops
                        {
                            return Err(ValidationError::OperandRequiresType {
                                operand: Id::try_from(*operand_id)
                                    .unwrap_or(Id::try_from(1u32).unwrap()),
                            }
                            .into());
                        }

                        // Check: Non-semantic result cannot be used in semantic instruction
                        if is_semantic && is_non_semantic_instruction(def_inst, ctx) {
                            return Err(ValidationError::NonSemanticUsedInSemantic {
                                operand: Id::try_from(*operand_id)
                                    .unwrap_or(Id::try_from(1u32).unwrap()),
                            }
                            .into());
                        }
                    }
                }
            }

            // Check OpExtInstWithForwardRefsKHR specific requirements
            if opcode == Op::ExtInstWithForwardRefsKHR {
                // Must be a non-semantic instruction
                if !is_non_semantic_instruction(inst, ctx) {
                    return Err(ValidationError::ExtInstWithForwardRefsNotNonSemantic.into());
                }

                // Must have at least one forward reference
                // (This is hard to check without tracking forward declarations,
                // so we'll skip this for now - the C++ version tracks this during parsing)
            }
        }

        Ok(())
    }
}

/// Returns all general type validation rules.
pub fn all_general_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &ReservedOpcodeRule,
        &TypeUniquenessRule,
        &ResultTypesAreTypesRule,
        &TypeFunctionsRule,
        &OperandDefinitionsRule,
        &IdPassRule,
    ]
}
