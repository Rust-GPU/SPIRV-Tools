//! Miscellaneous instruction validation rules.
//!
//! This module validates SPIR-V miscellaneous instructions including:
//!
//! - OpUndef - undefined value creation
//! - OpReadClockKHR - shader clock operations
//! - OpAssumeTrueKHR - assumption hints
//! - OpExpectKHR - expectation hints
//! - OpBeginInvocationInterlockEXT / OpEndInvocationInterlockEXT - interlock operations
//! - OpDemoteToHelperInvocationEXT / OpIsHelperInvocationEXT - helper invocations

use rspirv::dr::Operand;
use rspirv::spirv::{Capability, ExecutionMode, ExecutionModel, Op, Scope};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::get_type_structure;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId, TypeId, TypeStructure};
use crate::validation::ValidationResult;

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if a type contains 8- or 16-bit integers or floats that are restricted.
fn contains_limited_use_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let has_int8 = ctx.has_capability(Capability::Int8);
    let has_int16 = ctx.has_capability(Capability::Int16);
    let has_float16 = ctx.has_capability(Capability::Float16);

    contains_limited_type_recursive(type_id, ctx, has_int8, has_int16, has_float16)
}

fn contains_limited_type_recursive(
    type_id: TypeId,
    ctx: &ValidationContext<'_>,
    has_int8: bool,
    has_int16: bool,
    has_float16: bool,
) -> bool {
    use crate::validation::types::ScalarKind;

    let ty = get_type_structure(type_id, ctx.definitions);
    match ty {
        TypeStructure::Scalar(ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w)) => {
            (!has_int8 && w.get() == 8) || (!has_int16 && w.get() == 16)
        }
        TypeStructure::Scalar(ScalarKind::Float(w)) => !has_float16 && w.get() == 16,
        TypeStructure::Vector { component, .. } => match component {
            ScalarKind::SignedInt(w) | ScalarKind::UnsignedInt(w) => {
                (!has_int8 && w.get() == 8) || (!has_int16 && w.get() == 16)
            }
            ScalarKind::Float(w) => !has_float16 && w.get() == 16,
            _ => false,
        },
        TypeStructure::Array { element, .. } => {
            contains_limited_type_recursive(element, ctx, has_int8, has_int16, has_float16)
        }
        TypeStructure::RuntimeArray { element } => {
            contains_limited_type_recursive(element, ctx, has_int8, has_int16, has_float16)
        }
        TypeStructure::Struct { members } => members
            .iter()
            .any(|m| contains_limited_type_recursive(*m, ctx, has_int8, has_int16, has_float16)),
        TypeStructure::Pointer { pointee, .. } => {
            if let Some(p) = pointee {
                contains_limited_type_recursive(p, ctx, has_int8, has_int16, has_float16)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_pointer_type(type_id: TypeId, ctx: &ValidationContext<'_>) -> bool {
    let ty = get_type_structure(type_id, ctx.definitions);
    matches!(
        ty,
        TypeStructure::Pointer { .. } | TypeStructure::ForwardPointer { .. }
    )
}

/// Validates OpUndef instructions.
///
/// - Cannot create undefined values with void type
/// - Cannot create undefined values with 8- or 16-bit types (with Shader capability)
pub struct UndefRule;

impl ValidationRule for UndefRule {
    fn name(&self) -> &'static str {
        "undef"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        for inst in ctx.module.all_inst_iter() {
            if inst.class.opcode != Op::Undef {
                continue;
            }

            let Some(result_type_raw) = inst.result_type else {
                continue;
            };

            // Check for void type
            if let Ok(type_result_id) = ResultId::try_from(result_type_raw) {
                if let Some(type_inst) = ctx.definitions.get(&type_result_id) {
                    if type_inst.class.opcode == Op::TypeVoid {
                        return Err(ValidationError::UndefCannotBeVoid {
                            instruction_id: inst.result_id.map(to_id),
                        }
                        .into());
                    }
                }
            }

            // Check for restricted 8/16-bit types with Shader capability
            if ctx.has_capability(Capability::Shader) {
                if let Ok(type_id) = TypeId::try_from(result_type_raw) {
                    if !is_pointer_type(type_id, ctx) && contains_limited_use_type(type_id, ctx) {
                        return Err(ValidationError::UndefCannotBeSmallType {
                            instruction_id: inst.result_id.map(to_id),
                        }
                        .into());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpReadClockKHR instructions.
///
/// - Scope must be valid (Subgroup or Device for Vulkan, Workgroup/Subgroup/Device for OpenCL)
/// - Result type must be 64-bit unsigned int or vec2 of 32-bit unsigned int
pub struct ShaderClockRule;

impl ValidationRule for ShaderClockRule {
    fn name(&self) -> &'static str {
        "shader-clock"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ReadClockKHR {
                        continue;
                    }

                    // Check scope
                    if let Some(Operand::IdRef(scope_id)) = inst.operands.first() {
                        // Try to evaluate scope constant
                        if let Ok(scope_result_id) = ResultId::try_from(*scope_id) {
                            if let Some(scope_inst) = ctx.definitions.get(&scope_result_id) {
                                if scope_inst.class.opcode == Op::Constant {
                                    if let Some(Operand::LiteralBit32(scope_val)) =
                                        scope_inst.operands.first()
                                    {
                                        let scope = Scope::from_u32(*scope_val);
                                        if ctx.env.is_vulkan() {
                                            if scope != Some(Scope::Subgroup)
                                                && scope != Some(Scope::Device)
                                            {
                                                return Err(
                                                    ValidationError::ShaderClockInvalidScope {
                                                        function: func_id,
                                                        block: block_id,
                                                        expected: "Subgroup or Device",
                                                    }
                                                    .into(),
                                                );
                                            }
                                        } else if ctx.env.is_opencl()
                                            && scope != Some(Scope::Subgroup)
                                            && scope != Some(Scope::Workgroup)
                                            && scope != Some(Scope::Device)
                                        {
                                            return Err(ValidationError::ShaderClockInvalidScope {
                                                function: func_id,
                                                block: block_id,
                                                expected: "Subgroup, Workgroup, or Device",
                                            }
                                            .into());
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Check result type is 64-bit uint or vec2<u32>
                    if let Some(result_type_id) = inst.result_type {
                        // Could be: u64 scalar OR vec2<u32>
                        let bit_width = resolver.get_bit_width(result_type_id, ctx.definitions);
                        let dimension = resolver.get_dimension(result_type_id, ctx.definitions);
                        let is_unsigned = resolver
                            .is_unsigned_int_scalar_or_vector(result_type_id, ctx.definitions);

                        // Valid: u64 (dim=1, width=64) or uvec2<u32> (dim=2, width=32)
                        let is_valid = is_unsigned
                            && ((dimension == 1 && bit_width == Some(64))
                                || (dimension == 2 && bit_width == Some(32)));

                        if !is_valid {
                            return Err(ValidationError::ShaderClockInvalidResultType {
                                function: func_id,
                                block: block_id,
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

/// Validates OpAssumeTrueKHR instructions.
///
/// - Value operand must be a boolean scalar
pub struct AssumeTrueRule;

impl ValidationRule for AssumeTrueRule {
    fn name(&self) -> &'static str {
        "assume-true"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::AssumeTrueKHR {
                        continue;
                    }

                    if let Some(Operand::IdRef(value_id)) = inst.operands.first() {
                        if let Ok(value_result_id) = ResultId::try_from(*value_id) {
                            if let Some(value_inst) = ctx.definitions.get(&value_result_id) {
                                if let Some(value_type_id) = value_inst.result_type {
                                    if !resolver.is_bool_scalar(value_type_id, ctx.definitions) {
                                        return Err(ValidationError::AssumeTrueNotBool {
                                            function: func_id,
                                            block: block_id,
                                        }
                                        .into());
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

/// Validates OpExpectKHR instructions.
///
/// - Result type must be scalar/vector of int or bool
/// - Value and ExpectedValue must match result type
pub struct ExpectRule;

impl ValidationRule for ExpectRule {
    fn name(&self) -> &'static str {
        "expect"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::ExpectKHR {
                        continue;
                    }

                    let Some(result_type_id) = inst.result_type else {
                        continue;
                    };

                    // Result must be int or bool scalar/vector
                    let is_valid_result = resolver
                        .is_bool_scalar_or_vector(result_type_id, ctx.definitions)
                        || resolver.is_int_scalar_or_vector(result_type_id, ctx.definitions);

                    if !is_valid_result {
                        return Err(ValidationError::ExpectInvalidResultType {
                            function: func_id,
                            block: block_id,
                        }
                        .into());
                    }

                    // Check Value operand type matches
                    let get_operand_type = |idx: usize| -> Option<u32> {
                        let operand_id = inst.operands.get(idx).and_then(|op| match op {
                            Operand::IdRef(id) => Some(*id),
                            _ => None,
                        })?;
                        let operand_inst = ResultId::try_from(operand_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))?;
                        operand_inst.result_type
                    };

                    if let Some(value_type) = get_operand_type(0) {
                        if value_type != result_type_id {
                            return Err(ValidationError::ExpectValueTypeMismatch {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }

                    if let Some(expected_type) = get_operand_type(1) {
                        if expected_type != result_type_id {
                            return Err(ValidationError::ExpectExpectedValueTypeMismatch {
                                function: func_id,
                                block: block_id,
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

/// Validates OpIsHelperInvocationEXT instructions.
///
/// - Result type must be bool scalar
/// - Must be in Fragment execution model
pub struct IsHelperInvocationRule;

impl ValidationRule for IsHelperInvocationRule {
    fn name(&self) -> &'static str {
        "is-helper-invocation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let resolver = DefaultTypeResolver;

        // Check if any IsHelperInvocationEXT instructions exist
        let has_is_helper = ctx.module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions
                    .iter()
                    .any(|i| i.class.opcode == Op::IsHelperInvocationEXT)
            })
        });

        if !has_is_helper {
            return Ok(());
        }

        // Check execution model requirement
        let has_fragment = ctx.entry_models.contains(&ExecutionModel::Fragment);
        if !has_fragment && !ctx.entry_models.is_empty() {
            // Find an instruction to report in the error
            for func in &ctx.module.functions {
                let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);
                for block in &func.blocks {
                    let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);
                    for inst in &block.instructions {
                        if inst.class.opcode == Op::IsHelperInvocationEXT {
                            return Err(ValidationError::IsHelperInvocationRequiresFragment {
                                function: func_id,
                                block: block_id,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        for func in &ctx.module.functions {
            let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);

            for block in &func.blocks {
                let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);

                for inst in &block.instructions {
                    if inst.class.opcode != Op::IsHelperInvocationEXT {
                        continue;
                    }

                    // Result type must be bool scalar
                    if let Some(result_type_id) = inst.result_type {
                        if !resolver.is_bool_scalar(result_type_id, ctx.definitions) {
                            return Err(ValidationError::IsHelperInvocationNotBool {
                                function: func_id,
                                block: block_id,
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

/// Validates OpDemoteToHelperInvocationEXT instructions.
///
/// - Must be in Fragment execution model
pub struct DemoteToHelperInvocationRule;

impl ValidationRule for DemoteToHelperInvocationRule {
    fn name(&self) -> &'static str {
        "demote-to-helper-invocation"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Check if any DemoteToHelperInvocationEXT instructions exist
        let has_demote = ctx.module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions
                    .iter()
                    .any(|i| i.class.opcode == Op::DemoteToHelperInvocationEXT)
            })
        });

        if !has_demote {
            return Ok(());
        }

        // Check execution model requirement
        let has_fragment = ctx.entry_models.contains(&ExecutionModel::Fragment);
        if !has_fragment && !ctx.entry_models.is_empty() {
            // Find an instruction to report in the error
            for func in &ctx.module.functions {
                let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);
                for block in &func.blocks {
                    let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);
                    for inst in &block.instructions {
                        if inst.class.opcode == Op::DemoteToHelperInvocationEXT {
                            return Err(ValidationError::DemoteToHelperRequiresFragment {
                                function: func_id,
                                block: block_id,
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

/// Valid interlock execution modes.
const INTERLOCK_EXECUTION_MODES: &[ExecutionMode] = &[
    ExecutionMode::PixelInterlockOrderedEXT,
    ExecutionMode::PixelInterlockUnorderedEXT,
    ExecutionMode::SampleInterlockOrderedEXT,
    ExecutionMode::SampleInterlockUnorderedEXT,
    ExecutionMode::ShadingRateInterlockOrderedEXT,
    ExecutionMode::ShadingRateInterlockUnorderedEXT,
];

/// Validates OpBeginInvocationInterlockEXT and OpEndInvocationInterlockEXT instructions.
///
/// - Must be in Fragment execution model
/// - Must have an interlock execution mode declared
pub struct InvocationInterlockRule;

impl ValidationRule for InvocationInterlockRule {
    fn name(&self) -> &'static str {
        "invocation-interlock"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Check if any interlock instructions exist
        let has_interlock = ctx.module.functions.iter().any(|f| {
            f.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    matches!(
                        i.class.opcode,
                        Op::BeginInvocationInterlockEXT | Op::EndInvocationInterlockEXT
                    )
                })
            })
        });

        if !has_interlock {
            return Ok(());
        }

        // Check execution model requirement (must be Fragment)
        let has_fragment = ctx.entry_models.contains(&ExecutionModel::Fragment);
        if !has_fragment && !ctx.entry_models.is_empty() {
            // Find an instruction to report in the error
            for func in &ctx.module.functions {
                let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);
                for block in &func.blocks {
                    let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);
                    for inst in &block.instructions {
                        if matches!(
                            inst.class.opcode,
                            Op::BeginInvocationInterlockEXT | Op::EndInvocationInterlockEXT
                        ) {
                            return Err(ValidationError::InvocationInterlockRequiresFragment {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
                            }
                            .into());
                        }
                    }
                }
            }
        }

        // Check for required interlock execution mode
        let has_interlock_mode = ctx.module.execution_modes.iter().any(|mode_inst| {
            mode_inst.operands.get(1).is_some_and(|operand| {
                if let Operand::ExecutionMode(mode) = operand {
                    INTERLOCK_EXECUTION_MODES.contains(mode)
                } else {
                    false
                }
            })
        });

        if !has_interlock_mode {
            // Find an instruction to report in the error
            for func in &ctx.module.functions {
                let func_id = func.def.as_ref().and_then(|d| d.result_id).map(to_id);
                for block in &func.blocks {
                    let block_id = block.label.as_ref().and_then(|l| l.result_id).map(to_id);
                    for inst in &block.instructions {
                        if matches!(
                            inst.class.opcode,
                            Op::BeginInvocationInterlockEXT | Op::EndInvocationInterlockEXT
                        ) {
                            return Err(ValidationError::InvocationInterlockRequiresMode {
                                function: func_id,
                                block: block_id,
                                opcode: inst.class.opcode,
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

/// Returns all miscellaneous validation rules.
pub fn all_misc_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(UndefRule),
        Box::new(ShaderClockRule),
        Box::new(AssumeTrueRule),
        Box::new(ExpectRule),
        Box::new(IsHelperInvocationRule),
        Box::new(DemoteToHelperInvocationRule),
        Box::new(InvocationInterlockRule),
    ]
}
