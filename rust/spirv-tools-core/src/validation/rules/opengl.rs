//! OpenGL-specific validation rules.
//!
//! This module validates OpenGL-specific SPIR-V requirements including:
//!
//! - ARB_gl_spirv binding requirements for uniform/storage block variables

use rspirv::spirv::{Decoration, Op, StorageClass};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::helpers::is_opengl_env;
use crate::validation::rules::vulkan::{
    get_array_element_struct, get_variable_pointee_type, has_decoration,
};
use crate::validation::types::ResultId;
use crate::validation::ValidationResult;

// ============================================================================
// OpenGL Buffer Binding Rule
// ============================================================================

/// Validates that uniform and shader storage block variables have Binding decorations in OpenGL.
///
/// From ARB_gl_spirv extension:
/// Uniform and shader storage block variables must also be decorated with a *Binding*.
pub struct OpenGlBufferBindingRule;

impl ValidationRule for OpenGlBufferBindingRule {
    fn name(&self) -> &'static str {
        "opengl-buffer-binding"
    }

    fn should_skip(&self, ctx: &ValidationContext<'_>) -> bool {
        !is_opengl_env(ctx.env)
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        // Build a set of variable IDs referenced by any entry point
        let mut entry_point_vars: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        for ep in &ctx.module.entry_points {
            let mut operands = ep.operands.iter();
            // Skip ExecutionModel
            let _ = operands.next();
            // Skip entry point function ID
            let _ = operands.next();
            // Skip name (LiteralString)
            let _ = operands.next();
            // Remaining operands are interface variable IDs
            for operand in operands {
                if let rspirv::dr::Operand::IdRef(id) = operand {
                    entry_point_vars.insert(*id);
                }
            }
        }

        for inst in &ctx.module.types_global_values {
            if inst.class.opcode != Op::Variable && inst.class.opcode != Op::UntypedVariableKHR {
                continue;
            }

            let storage_class = match inst.operands.first() {
                Some(rspirv::dr::Operand::StorageClass(sc)) => *sc,
                _ => continue,
            };

            let var_id = match inst.result_id {
                Some(id) => id,
                None => continue,
            };

            let uniform = storage_class == StorageClass::Uniform
                || storage_class == StorageClass::UniformConstant;
            let storage_buffer = storage_class == StorageClass::StorageBuffer;

            if !uniform && !storage_buffer {
                continue;
            }

            // Get the pointee struct type
            let pointee_type_id = get_variable_pointee_type(inst, ctx);
            let Some(struct_id) = pointee_type_id else {
                continue;
            };

            // Check if pointee is a struct (possibly through arrays)
            let pointee_opcode = ResultId::try_from(struct_id)
                .ok()
                .and_then(|rid| ctx.opcodes.get(&rid))
                .copied();

            let final_struct_id = if matches!(
                pointee_opcode,
                Some(Op::TypeArray) | Some(Op::TypeRuntimeArray)
            ) {
                get_array_element_struct(struct_id, ctx)
            } else if pointee_opcode == Some(Op::TypeStruct) {
                Some(struct_id)
            } else {
                None
            };

            let Some(struct_id) = final_struct_id else {
                continue;
            };

            let has_block = has_decoration(ctx, struct_id, Decoration::Block);
            let has_buffer_block = has_decoration(ctx, struct_id, Decoration::BufferBlock);

            // OpenGL requires Binding on:
            // - Uniform variables with Block or BufferBlock decoration
            // - StorageBuffer variables with Block decoration
            if (uniform && (has_block || has_buffer_block))
                || (storage_buffer && has_block)
            {
                // Only check variables referenced by entry points
                if !entry_point_vars.is_empty()
                    && entry_point_vars.contains(&var_id)
                    && !has_decoration(ctx, var_id, Decoration::Binding)
                {
                    return Err(ValidationError::OpenGlBufferMissingBindingDecoration {
                        storage_class,
                        variable_id: var_id,
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// All OpenGL rules
// ============================================================================

/// Returns all OpenGL-specific validation rules.
pub fn all_opengl_rules() -> Vec<&'static dyn ValidationRule> {
    vec![&OpenGlBufferBindingRule]
}
