//! Primitive instruction validation rules.
//!
//! This module validates SPIR-V primitive instructions:
//!
//! - OpEmitVertex / OpEndPrimitive - require Geometry execution model
//! - OpEmitStreamVertex / OpEndStreamPrimitive - require Geometry execution model
//!   and Stream operand must be a constant int scalar

use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionModel, Op};

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::types::{Id, ResultId};

fn to_id(id: u32) -> Id {
    Id::try_from(id).unwrap_or_else(|_| Id::try_from(1u32).unwrap())
}

/// Check if an opcode is a constant instruction.
fn is_constant_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::Constant
            | Op::ConstantTrue
            | Op::ConstantFalse
            | Op::ConstantNull
            | Op::ConstantComposite
            | Op::ConstantSampler
            | Op::ConstantPipeStorage
            | Op::SpecConstant
            | Op::SpecConstantTrue
            | Op::SpecConstantFalse
            | Op::SpecConstantComposite
            | Op::SpecConstantOp
    )
}

/// Validates primitive instructions require Geometry execution model.
///
/// - OpEmitVertex, OpEndPrimitive, OpEmitStreamVertex, OpEndStreamPrimitive
///   all require Geometry execution model
pub struct PrimitiveExecutionModelRule;

impl ValidationRule for PrimitiveExecutionModelRule {
    fn name(&self) -> &'static str {
        "primitive-execution-model"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        // Check if we have Geometry execution model
        let has_geometry = ctx.module.entry_points.iter().any(|ep| {
            ep.operands.first().map_or(false, |op| {
                matches!(op, Operand::ExecutionModel(ExecutionModel::Geometry))
            })
        });

        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id);

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id);

                for inst in &block.instructions {
                    let requires_geometry = matches!(
                        inst.class.opcode,
                        Op::EmitVertex
                            | Op::EndPrimitive
                            | Op::EmitStreamVertex
                            | Op::EndStreamPrimitive
                    );

                    if requires_geometry && !has_geometry {
                        return Err(ValidationError::PrimitiveRequiresGeometry {
                            function: func_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates stream primitive instructions have valid Stream operands.
///
/// - OpEmitStreamVertex and OpEndStreamPrimitive require Stream to be:
///   - An int scalar type
///   - A constant instruction
pub struct StreamPrimitiveRule;

impl ValidationRule for StreamPrimitiveRule {
    fn name(&self) -> &'static str {
        "stream-primitive"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        for func in &ctx.module.functions {
            let func_id = func
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .map(to_id);

            for block in &func.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .map(to_id);

                for inst in &block.instructions {
                    if !matches!(
                        inst.class.opcode,
                        Op::EmitStreamVertex | Op::EndStreamPrimitive
                    ) {
                        continue;
                    }

                    // Get Stream operand (first operand)
                    let Some(Operand::IdRef(stream_id)) = inst.operands.first() else {
                        continue;
                    };

                    let Some(stream_inst) = ResultId::try_from(*stream_id)
                        .ok()
                        .and_then(|rid| ctx.definitions.get(&rid))
                    else {
                        continue;
                    };

                    // Check type is int scalar
                    if let Some(stream_type_id) = stream_inst.result_type {
                        if let Some(type_inst) = ResultId::try_from(stream_type_id)
                            .ok()
                            .and_then(|rid| ctx.definitions.get(&rid))
                        {
                            if type_inst.class.opcode != Op::TypeInt {
                                return Err(ValidationError::StreamNotIntScalar {
                                    function: func_id,
                                    block: block_id,
                                    opcode: inst.class.opcode,
                                });
                            }
                        }
                    }

                    // Check Stream is a constant instruction
                    if !is_constant_opcode(stream_inst.class.opcode) {
                        return Err(ValidationError::StreamNotConstant {
                            function: func_id,
                            block: block_id,
                            opcode: inst.class.opcode,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Returns all primitive validation rules.
pub fn all_primitive_rules() -> Vec<Box<dyn ValidationRule>> {
    vec![
        Box::new(PrimitiveExecutionModelRule),
        Box::new(StreamPrimitiveRule),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_primitive_rules() {
        let rules = all_primitive_rules();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name(), "primitive-execution-model");
        assert_eq!(rules[1].name(), "stream-primitive");
    }

    #[test]
    fn test_is_constant_opcode() {
        assert!(is_constant_opcode(Op::Constant));
        assert!(is_constant_opcode(Op::ConstantTrue));
        assert!(is_constant_opcode(Op::ConstantFalse));
        assert!(is_constant_opcode(Op::ConstantNull));
        assert!(is_constant_opcode(Op::ConstantComposite));
        assert!(is_constant_opcode(Op::SpecConstant));
        assert!(is_constant_opcode(Op::SpecConstantTrue));
        assert!(is_constant_opcode(Op::SpecConstantFalse));

        assert!(!is_constant_opcode(Op::Variable));
        assert!(!is_constant_opcode(Op::FunctionParameter));
        assert!(!is_constant_opcode(Op::Load));
    }
}
