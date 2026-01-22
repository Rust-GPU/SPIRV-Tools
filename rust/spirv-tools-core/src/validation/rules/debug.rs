//! Debug instruction validation rules.
//!
//! This module implements validation for SPIR-V debug instructions including:
//! - OpMemberName (struct member naming)
//! - OpLine (source line information)

use rspirv::dr::Operand;
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};
use crate::validation::ValidationResult;

/// Helper to convert a u32 to Id (with fallback to id 1).
fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Validates OpMemberName instruction.
///
/// Validation rules:
/// - Type operand must be a struct type (OpTypeStruct)
/// - Member index must be within the struct's member count
pub struct MemberNameRule;

impl ValidationRule for MemberNameRule {
    fn name(&self) -> &'static str {
        "member-name"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // OpMemberName is in debug names section
        for inst in &module.debug_names {
            if inst.class.opcode == Op::MemberName {
                // Operand 0: Type (must be struct)
                // Operand 1: Member index
                // Operand 2: Name (string)

                if let (Some(Operand::IdRef(type_id)), Some(member_operand)) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    // Check that type is a struct
                    if let Ok(result_id) = ResultId::try_from(*type_id) {
                        if let Some(type_inst) = ctx.definitions.get(&result_id) {
                            if type_inst.class.opcode != Op::TypeStruct {
                                return Err(ValidationError::DebugMemberNameNotStruct {
                                    type_id: to_id(*type_id),
                                }
                                .into());
                            }

                            // Get the member count from the struct type
                            // OpTypeStruct has member types as operands, so operands.len() = member count
                            let member_count = type_inst.operands.len() as u32;

                            // Get the member index
                            if let Operand::LiteralBit32(member_index) = member_operand {
                                if *member_index >= member_count {
                                    return Err(ValidationError::DebugMemberNameIndexOutOfBounds {
                                        type_id: to_id(*type_id),
                                        member_index: *member_index,
                                        member_count,
                                    }
                                    .into());
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

/// Validates OpLine instruction.
///
/// Validation rules:
/// - Target operand (File) must be an OpString
pub struct LineRule;

impl ValidationRule for LineRule {
    fn name(&self) -> &'static str {
        "line"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
        let module = ctx.module();

        // Helper to validate a single OpLine
        let validate_line = |inst: &rspirv::dr::Instruction| -> ValidationResult {
            if inst.class.opcode == Op::Line {
                // Operand 0: File (must be OpString)
                if let Some(Operand::IdRef(file_id)) = inst.operands.first() {
                    if let Ok(result_id) = ResultId::try_from(*file_id) {
                        if let Some(file_inst) = ctx.definitions.get(&result_id) {
                            if file_inst.class.opcode != Op::String {
                                return Err(ValidationError::DebugLineTargetNotString {
                                    file_id: to_id(*file_id),
                                }
                                .into());
                            }
                        }
                    }
                }
            }
            Ok(())
        };

        // Check debug source section
        for inst in &module.debug_string_source {
            validate_line(inst)?;
        }

        // Check functions (OpLine can appear before instructions)
        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    validate_line(inst)?;
                }
            }
        }

        Ok(())
    }
}

/// Returns all debug instruction validation rules.
pub fn all_debug_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![Box::new(MemberNameRule), Box::new(LineRule)]
}
