//! Extended instruction validation rules.
//!
//! This module validates SPIR-V extended instructions including:
//!
//! - GLSL.std.450 extended instruction set
//! - OpenCL.std extended instruction set (partial)
//! - Extended instruction import validation

use std::collections::HashMap;

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId};

// ============================================================================
// GLSL.std.450 Opcode Constants
// ============================================================================

/// GLSL.std.450 opcodes for floating-point operations.
mod glsl {
    // Common floating-point
    pub const ROUND: u32 = 1;
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const FABS: u32 = 4;
    pub const SABS: u32 = 5;
    pub const FSIGN: u32 = 6;
    pub const SSIGN: u32 = 7;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;

    // Angle and trigonometric
    pub const RADIANS: u32 = 11;
    pub const DEGREES: u32 = 12;
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TAN: u32 = 15;
    pub const ASIN: u32 = 16;
    pub const ACOS: u32 = 17;
    pub const ATAN: u32 = 18;
    pub const SINH: u32 = 19;
    pub const COSH: u32 = 20;
    pub const TANH: u32 = 21;
    pub const ASINH: u32 = 22;
    pub const ACOSH: u32 = 23;
    pub const ATANH: u32 = 24;
    pub const ATAN2: u32 = 25;

    // Exponential
    pub const POW: u32 = 26;
    pub const EXP: u32 = 27;
    pub const LOG: u32 = 28;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;

    // Matrix
    pub const DETERMINANT: u32 = 33;
    pub const MATRIX_INVERSE: u32 = 34;

    // Modf/Frexp
    pub const MODF_STRUCT: u32 = 35;
    pub const MODF: u32 = 36;

    // Min/Max/Clamp
    pub const FMIN: u32 = 37;
    pub const UMIN: u32 = 38;
    pub const SMIN: u32 = 39;
    pub const FMAX: u32 = 40;
    pub const UMAX: u32 = 41;
    pub const SMAX: u32 = 42;
    pub const FCLAMP: u32 = 43;
    pub const UCLAMP: u32 = 44;
    pub const SCLAMP: u32 = 45;

    // Mix/Step
    pub const FMIX: u32 = 46;
    pub const IMIX: u32 = 47;
    pub const STEP: u32 = 48;
    pub const SMOOTH_STEP: u32 = 49;

    // FMA
    pub const FMA: u32 = 50;

    // Frexp/Ldexp
    pub const FREXP_STRUCT: u32 = 51;
    pub const FREXP: u32 = 52;
    pub const LDEXP: u32 = 53;

    // Pack/Unpack
    pub const PACK_SNORM4X8: u32 = 54;
    pub const PACK_UNORM4X8: u32 = 55;
    pub const PACK_SNORM2X16: u32 = 56;
    pub const PACK_UNORM2X16: u32 = 57;
    pub const PACK_HALF2X16: u32 = 58;
    pub const PACK_DOUBLE2X32: u32 = 59;
    pub const UNPACK_SNORM2X16: u32 = 60;
    pub const UNPACK_UNORM2X16: u32 = 61;
    pub const UNPACK_HALF2X16: u32 = 62;
    pub const UNPACK_SNORM4X8: u32 = 63;
    pub const UNPACK_UNORM4X8: u32 = 64;
    pub const UNPACK_DOUBLE2X32: u32 = 65;

    // Geometry
    pub const LENGTH: u32 = 66;
    pub const DISTANCE: u32 = 67;
    pub const CROSS: u32 = 68;
    pub const NORMALIZE: u32 = 69;
    pub const FACE_FORWARD: u32 = 70;
    pub const REFLECT: u32 = 71;
    pub const REFRACT: u32 = 72;

    // Integer
    pub const FIND_ILSB: u32 = 73;
    pub const FIND_SMSB: u32 = 74;
    pub const FIND_UMSB: u32 = 75;

    // Interpolate
    pub const INTERPOLATE_AT_CENTROID: u32 = 76;
    pub const INTERPOLATE_AT_SAMPLE: u32 = 77;
    pub const INTERPOLATE_AT_OFFSET: u32 = 78;

    // NaN-aware min/max/clamp
    pub const NMIN: u32 = 79;
    pub const NMAX: u32 = 80;
    pub const NCLAMP: u32 = 81;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the GLSL.std.450 import ID from the module.
fn get_glsl_import_id(ctx: &ValidationContext<'_>) -> Option<u32> {
    for inst in &ctx.module.ext_inst_imports {
        if inst.class.opcode == Op::ExtInstImport {
            if let Some(Operand::LiteralString(name)) = inst.operands.first() {
                if name == "GLSL.std.450" {
                    return inst.result_id;
                }
            }
        }
    }
    None
}

/// Check if an instruction is a GLSL.std.450 extended instruction.
fn is_glsl_ext_inst(inst: &Instruction, glsl_import_id: u32) -> bool {
    if inst.class.opcode != Op::ExtInst {
        return false;
    }
    // Operand 0 is the extension set ID
    if let Some(Operand::IdRef(ext_set)) = inst.operands.first() {
        return *ext_set == glsl_import_id;
    }
    false
}

/// Get the GLSL.std.450 opcode from an OpExtInst instruction.
fn get_glsl_opcode(inst: &Instruction) -> Option<u32> {
    // Operand 1 is the instruction number (LiteralExtInstInteger)
    if let Some(Operand::LiteralExtInstInteger(opcode)) = inst.operands.get(1) {
        return Some(*opcode);
    }
    None
}

/// Get the name of a GLSL.std.450 instruction.
fn get_glsl_name(opcode: u32) -> &'static str {
    match opcode {
        glsl::ROUND => "Round",
        glsl::ROUND_EVEN => "RoundEven",
        glsl::TRUNC => "Trunc",
        glsl::FABS => "FAbs",
        glsl::SABS => "SAbs",
        glsl::FSIGN => "FSign",
        glsl::SSIGN => "SSign",
        glsl::FLOOR => "Floor",
        glsl::CEIL => "Ceil",
        glsl::FRACT => "Fract",
        glsl::RADIANS => "Radians",
        glsl::DEGREES => "Degrees",
        glsl::SIN => "Sin",
        glsl::COS => "Cos",
        glsl::TAN => "Tan",
        glsl::ASIN => "Asin",
        glsl::ACOS => "Acos",
        glsl::ATAN => "Atan",
        glsl::SINH => "Sinh",
        glsl::COSH => "Cosh",
        glsl::TANH => "Tanh",
        glsl::ASINH => "Asinh",
        glsl::ACOSH => "Acosh",
        glsl::ATANH => "Atanh",
        glsl::ATAN2 => "Atan2",
        glsl::POW => "Pow",
        glsl::EXP => "Exp",
        glsl::LOG => "Log",
        glsl::EXP2 => "Exp2",
        glsl::LOG2 => "Log2",
        glsl::SQRT => "Sqrt",
        glsl::INVERSE_SQRT => "InverseSqrt",
        glsl::DETERMINANT => "Determinant",
        glsl::MATRIX_INVERSE => "MatrixInverse",
        glsl::MODF_STRUCT => "ModfStruct",
        glsl::MODF => "Modf",
        glsl::FMIN => "FMin",
        glsl::UMIN => "UMin",
        glsl::SMIN => "SMin",
        glsl::FMAX => "FMax",
        glsl::UMAX => "UMax",
        glsl::SMAX => "SMax",
        glsl::FCLAMP => "FClamp",
        glsl::UCLAMP => "UClamp",
        glsl::SCLAMP => "SClamp",
        glsl::FMIX => "FMix",
        glsl::IMIX => "IMix",
        glsl::STEP => "Step",
        glsl::SMOOTH_STEP => "SmoothStep",
        glsl::FMA => "Fma",
        glsl::FREXP_STRUCT => "FrexpStruct",
        glsl::FREXP => "Frexp",
        glsl::LDEXP => "Ldexp",
        glsl::PACK_SNORM4X8 => "PackSnorm4x8",
        glsl::PACK_UNORM4X8 => "PackUnorm4x8",
        glsl::PACK_SNORM2X16 => "PackSnorm2x16",
        glsl::PACK_UNORM2X16 => "PackUnorm2x16",
        glsl::PACK_HALF2X16 => "PackHalf2x16",
        glsl::PACK_DOUBLE2X32 => "PackDouble2x32",
        glsl::UNPACK_SNORM2X16 => "UnpackSnorm2x16",
        glsl::UNPACK_UNORM2X16 => "UnpackUnorm2x16",
        glsl::UNPACK_HALF2X16 => "UnpackHalf2x16",
        glsl::UNPACK_SNORM4X8 => "UnpackSnorm4x8",
        glsl::UNPACK_UNORM4X8 => "UnpackUnorm4x8",
        glsl::UNPACK_DOUBLE2X32 => "UnpackDouble2x32",
        glsl::LENGTH => "Length",
        glsl::DISTANCE => "Distance",
        glsl::CROSS => "Cross",
        glsl::NORMALIZE => "Normalize",
        glsl::FACE_FORWARD => "FaceForward",
        glsl::REFLECT => "Reflect",
        glsl::REFRACT => "Refract",
        glsl::FIND_ILSB => "FindILsb",
        glsl::FIND_SMSB => "FindSMsb",
        glsl::FIND_UMSB => "FindUMsb",
        glsl::INTERPOLATE_AT_CENTROID => "InterpolateAtCentroid",
        glsl::INTERPOLATE_AT_SAMPLE => "InterpolateAtSample",
        glsl::INTERPOLATE_AT_OFFSET => "InterpolateAtOffset",
        glsl::NMIN => "NMin",
        glsl::NMAX => "NMax",
        glsl::NCLAMP => "NClamp",
        _ => "Unknown",
    }
}

// ============================================================================
// GLSL Float Operations Rule
// ============================================================================

/// Validates GLSL.std.450 floating-point operations.
///
/// These operations require:
/// - Result type is float scalar or vector
/// - All operands have the same type as the result
pub struct GlslFloatOpsRule;

impl ValidationRule for GlslFloatOpsRule {
    fn name(&self) -> &'static str {
        "glsl-float-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let glsl_import_id = match get_glsl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No GLSL.std.450 import
        };

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_glsl_ext_inst(inst, glsl_import_id) {
                        continue;
                    }

                    let opcode = match get_glsl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    // Float operations that require float result and matching operands
                    let is_float_op = matches!(
                        opcode,
                        glsl::ROUND
                            | glsl::ROUND_EVEN
                            | glsl::FABS
                            | glsl::TRUNC
                            | glsl::FSIGN
                            | glsl::FLOOR
                            | glsl::CEIL
                            | glsl::FRACT
                            | glsl::SQRT
                            | glsl::INVERSE_SQRT
                            | glsl::FMIN
                            | glsl::FMAX
                            | glsl::FCLAMP
                            | glsl::FMIX
                            | glsl::STEP
                            | glsl::SMOOTH_STEP
                            | glsl::FMA
                            | glsl::NORMALIZE
                            | glsl::FACE_FORWARD
                            | glsl::REFLECT
                            | glsl::NMIN
                            | glsl::NMAX
                            | glsl::NCLAMP
                    );

                    if !is_float_op {
                        continue;
                    }

                    // Validate result type is float scalar or vector
                    if let Some(result_type) = inst.result_type {
                        if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::ExtInstResultTypeMustBeFloat {
                                function: function_id,
                                block: block_id,
                                ext_inst_name: get_glsl_name(opcode),
                            });
                        }

                        // Validate all operands match result type
                        // Operands start at index 2 (after ext set ID and instruction number)
                        for operand in inst.operands.iter().skip(2) {
                            if let Operand::IdRef(operand_id) = operand {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if operand_type != result_type {
                                                return Err(
                                                    ValidationError::ExtInstOperandTypeMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
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

// ============================================================================
// GLSL Integer Operations Rule
// ============================================================================

/// Validates GLSL.std.450 integer operations.
///
/// These operations require:
/// - Result type is int scalar or vector
/// - All operands have matching dimensions and bit widths
pub struct GlslIntOpsRule;

impl ValidationRule for GlslIntOpsRule {
    fn name(&self) -> &'static str {
        "glsl-int-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let glsl_import_id = match get_glsl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No GLSL.std.450 import
        };

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_glsl_ext_inst(inst, glsl_import_id) {
                        continue;
                    }

                    let opcode = match get_glsl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    // Integer operations
                    let is_int_op = matches!(
                        opcode,
                        glsl::SABS
                            | glsl::SSIGN
                            | glsl::UMIN
                            | glsl::SMIN
                            | glsl::UMAX
                            | glsl::SMAX
                            | glsl::UCLAMP
                            | glsl::SCLAMP
                            | glsl::FIND_ILSB
                            | glsl::FIND_UMSB
                            | glsl::FIND_SMSB
                    );

                    if !is_int_op {
                        continue;
                    }

                    // Validate result type is int scalar or vector
                    if let Some(result_type) = inst.result_type {
                        if !resolver.is_int_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::ExtInstResultTypeMustBeInt {
                                function: function_id,
                                block: block_id,
                                ext_inst_name: get_glsl_name(opcode),
                            });
                        }

                        // For FindUMsb and FindSMsb, bit width must be 32
                        if opcode == glsl::FIND_UMSB || opcode == glsl::FIND_SMSB {
                            if let Ok(type_id) = ResultId::try_from(result_type) {
                                if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                    let bit_width = get_int_bit_width(type_inst, ctx.definitions);
                                    if bit_width != Some(32) {
                                        return Err(ValidationError::ExtInstRequires32BitInt {
                                            function: function_id,
                                            block: block_id,
                                            ext_inst_name: get_glsl_name(opcode),
                                        });
                                    }
                                }
                            }
                        }

                        // Validate all operands are int and match dimensions/bit width
                        for operand in inst.operands.iter().skip(2) {
                            if let Operand::IdRef(operand_id) = operand {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !resolver
                                                .is_int_scalar_or_vector(operand_type, ctx.definitions)
                                            {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeInt {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
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

/// Get the bit width of an integer type (scalar or vector component).
fn get_int_bit_width(
    type_inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    match type_inst.class.opcode {
        Op::TypeInt => {
            // OpTypeInt: operand 0 is width
            if let Some(Operand::LiteralBit32(width)) = type_inst.operands.first() {
                return Some(*width);
            }
        }
        Op::TypeVector => {
            // OpTypeVector: operand 0 is component type
            if let Some(Operand::IdRef(component_type_id)) = type_inst.operands.first() {
                if let Ok(component_result_id) = ResultId::try_from(*component_type_id) {
                    if let Some(component_inst) = definitions.get(&component_result_id) {
                        return get_int_bit_width(component_inst, definitions);
                    }
                }
            }
        }
        _ => {}
    }
    None
}

// ============================================================================
// GLSL Trigonometric Operations Rule
// ============================================================================

/// Validates GLSL.std.450 trigonometric operations.
///
/// These operations require:
/// - Result type is 16 or 32-bit float scalar or vector
/// - All operands match the result type
pub struct GlslTrigOpsRule;

impl ValidationRule for GlslTrigOpsRule {
    fn name(&self) -> &'static str {
        "glsl-trig-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let glsl_import_id = match get_glsl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No GLSL.std.450 import
        };

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_glsl_ext_inst(inst, glsl_import_id) {
                        continue;
                    }

                    let opcode = match get_glsl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    // Trigonometric and exponential operations with 16/32-bit restriction
                    let is_trig_op = matches!(
                        opcode,
                        glsl::RADIANS
                            | glsl::DEGREES
                            | glsl::SIN
                            | glsl::COS
                            | glsl::TAN
                            | glsl::ASIN
                            | glsl::ACOS
                            | glsl::ATAN
                            | glsl::SINH
                            | glsl::COSH
                            | glsl::TANH
                            | glsl::ASINH
                            | glsl::ACOSH
                            | glsl::ATANH
                            | glsl::EXP
                            | glsl::EXP2
                            | glsl::LOG
                            | glsl::LOG2
                            | glsl::ATAN2
                            | glsl::POW
                    );

                    if !is_trig_op {
                        continue;
                    }

                    // Validate result type is float scalar or vector
                    if let Some(result_type) = inst.result_type {
                        if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::ExtInstResultTypeMustBeFloat {
                                function: function_id,
                                block: block_id,
                                ext_inst_name: get_glsl_name(opcode),
                            });
                        }

                        // Validate bit width is 16 or 32
                        if let Ok(type_id) = ResultId::try_from(result_type) {
                            if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                let bit_width = get_float_bit_width(type_inst, ctx.definitions);
                                if bit_width != Some(16) && bit_width != Some(32) {
                                    return Err(ValidationError::ExtInstRequires16Or32BitFloat {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }
                        }

                        // Validate all operands match result type
                        for operand in inst.operands.iter().skip(2) {
                            if let Operand::IdRef(operand_id) = operand {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if operand_type != result_type {
                                                return Err(
                                                    ValidationError::ExtInstOperandTypeMismatch {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
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

/// Get the bit width of a float type (scalar or vector component).
fn get_float_bit_width(
    type_inst: &Instruction,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    match type_inst.class.opcode {
        Op::TypeFloat => {
            // OpTypeFloat: operand 0 is width
            if let Some(Operand::LiteralBit32(width)) = type_inst.operands.first() {
                return Some(*width);
            }
        }
        Op::TypeVector => {
            // OpTypeVector: operand 0 is component type
            if let Some(Operand::IdRef(component_type_id)) = type_inst.operands.first() {
                if let Ok(component_result_id) = ResultId::try_from(*component_type_id) {
                    if let Some(component_inst) = definitions.get(&component_result_id) {
                        return get_float_bit_width(component_inst, definitions);
                    }
                }
            }
        }
        _ => {}
    }
    None
}

// ============================================================================
// GLSL Pack/Unpack Operations Rule
// ============================================================================

/// Validates GLSL.std.450 pack/unpack operations.
pub struct GlslPackUnpackRule;

impl ValidationRule for GlslPackUnpackRule {
    fn name(&self) -> &'static str {
        "glsl-pack-unpack"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let glsl_import_id = match get_glsl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No GLSL.std.450 import
        };

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_glsl_ext_inst(inst, glsl_import_id) {
                        continue;
                    }

                    let opcode = match get_glsl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    match opcode {
                        // PackSnorm4x8, PackUnorm4x8 - result is 32-bit uint, operand is vec4 float
                        glsl::PACK_SNORM4X8 | glsl::PACK_UNORM4X8 => {
                            if let Some(result_type) = inst.result_type {
                                // Result must be 32-bit unsigned int scalar
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeInt {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }

                            // Operand must be vec4 float
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(2) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !is_float_vec4(operand_type, ctx) {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeVec4Float {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // PackSnorm2x16, PackUnorm2x16, PackHalf2x16 - result is 32-bit uint, operand is vec2 float
                        glsl::PACK_SNORM2X16 | glsl::PACK_UNORM2X16 | glsl::PACK_HALF2X16 => {
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_int_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeInt {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }

                            // Operand must be vec2 float
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(2) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !is_float_vec2(operand_type, ctx) {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeVec2Float {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // UnpackSnorm4x8, UnpackUnorm4x8 - result is vec4 float, operand is 32-bit uint
                        glsl::UNPACK_SNORM4X8 | glsl::UNPACK_UNORM4X8 => {
                            if let Some(result_type) = inst.result_type {
                                if !is_float_vec4(result_type, ctx) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeVec4Float {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }

                            // Operand must be 32-bit uint
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(2) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !resolver
                                                .is_int_scalar(operand_type, ctx.definitions)
                                            {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeInt {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // UnpackSnorm2x16, UnpackUnorm2x16, UnpackHalf2x16 - result is vec2 float, operand is 32-bit uint
                        glsl::UNPACK_SNORM2X16 | glsl::UNPACK_UNORM2X16 | glsl::UNPACK_HALF2X16 => {
                            if let Some(result_type) = inst.result_type {
                                if !is_float_vec2(result_type, ctx) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeVec2Float {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }

                            // Operand must be 32-bit uint
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(2) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !resolver
                                                .is_int_scalar(operand_type, ctx.definitions)
                                            {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeInt {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
                                                    },
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

/// Check if type is vec4 of float.
fn is_float_vec4(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = ctx.definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypeVector {
                // Check component count is 4
                if let Some(Operand::LiteralBit32(count)) = type_inst.operands.get(1) {
                    if *count != 4 {
                        return false;
                    }
                }
                // Check component type is float
                if let Some(Operand::IdRef(component_type_id)) = type_inst.operands.first() {
                    if let Ok(component_result_id) = ResultId::try_from(*component_type_id) {
                        if let Some(component_inst) = ctx.definitions.get(&component_result_id) {
                            return component_inst.class.opcode == Op::TypeFloat;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Check if type is vec2 of float.
fn is_float_vec2(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = ctx.definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypeVector {
                // Check component count is 2
                if let Some(Operand::LiteralBit32(count)) = type_inst.operands.get(1) {
                    if *count != 2 {
                        return false;
                    }
                }
                // Check component type is float
                if let Some(Operand::IdRef(component_type_id)) = type_inst.operands.first() {
                    if let Ok(component_result_id) = ResultId::try_from(*component_type_id) {
                        if let Some(component_inst) = ctx.definitions.get(&component_result_id) {
                            return component_inst.class.opcode == Op::TypeFloat;
                        }
                    }
                }
            }
        }
    }
    false
}

// ============================================================================
// GLSL Geometry Operations Rule
// ============================================================================

/// Validates GLSL.std.450 geometry operations (Length, Distance, Cross, etc.).
pub struct GlslGeometryOpsRule;

impl ValidationRule for GlslGeometryOpsRule {
    fn name(&self) -> &'static str {
        "glsl-geometry-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let glsl_import_id = match get_glsl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No GLSL.std.450 import
        };

        let resolver = DefaultTypeResolver;

        for function in &ctx.module.functions {
            let function_id = function
                .def
                .as_ref()
                .and_then(|d| d.result_id)
                .and_then(|id| Id::try_from(id).ok());

            for block in &function.blocks {
                let block_id = block
                    .label
                    .as_ref()
                    .and_then(|l| l.result_id)
                    .and_then(|id| Id::try_from(id).ok());

                for inst in &block.instructions {
                    if !is_glsl_ext_inst(inst, glsl_import_id) {
                        continue;
                    }

                    let opcode = match get_glsl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    match opcode {
                        // Length - result is float scalar, operand is float scalar or vector
                        glsl::LENGTH => {
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_float_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeFloatScalar {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }

                            // Operand must be float scalar or vector
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(2) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if !resolver.is_float_scalar_or_vector(
                                                operand_type,
                                                ctx.definitions,
                                            ) {
                                                return Err(
                                                    ValidationError::ExtInstOperandMustBeFloat {
                                                        function: function_id,
                                                        block: block_id,
                                                        ext_inst_name: get_glsl_name(opcode),
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Distance - result is float scalar, operands are float scalar or vector
                        glsl::DISTANCE => {
                            if let Some(result_type) = inst.result_type {
                                if !resolver.is_float_scalar(result_type, ctx.definitions) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeFloatScalar {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }
                        }

                        // Cross - result and operands must be vec3 float
                        glsl::CROSS => {
                            if let Some(result_type) = inst.result_type {
                                if !is_float_vec3(result_type, ctx) {
                                    return Err(ValidationError::ExtInstResultTypeMustBeVec3Float {
                                        function: function_id,
                                        block: block_id,
                                        ext_inst_name: get_glsl_name(opcode),
                                    });
                                }
                            }
                        }

                        // Refract - eta operand must be float scalar
                        glsl::REFRACT => {
                            // Third operand (index 4) is eta, must be float scalar
                            if let Some(Operand::IdRef(eta_id)) = inst.operands.get(4) {
                                if let Ok(eta_result_id) = ResultId::try_from(*eta_id) {
                                    if let Some(eta_inst) = ctx.definitions.get(&eta_result_id) {
                                        if let Some(eta_type) = eta_inst.result_type {
                                            if !resolver.is_float_scalar(eta_type, ctx.definitions)
                                            {
                                                return Err(
                                                    ValidationError::ExtInstEtaMustBeFloatScalar {
                                                        function: function_id,
                                                        block: block_id,
                                                    },
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

/// Check if type is vec3 of float.
fn is_float_vec3(type_id: u32, ctx: &ValidationContext<'_>) -> bool {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = ctx.definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypeVector {
                // Check component count is 3
                if let Some(Operand::LiteralBit32(count)) = type_inst.operands.get(1) {
                    if *count != 3 {
                        return false;
                    }
                }
                // Check component type is float
                if let Some(Operand::IdRef(component_type_id)) = type_inst.operands.first() {
                    if let Ok(component_result_id) = ResultId::try_from(*component_type_id) {
                        if let Some(component_inst) = ctx.definitions.get(&component_result_id) {
                            return component_inst.class.opcode == Op::TypeFloat;
                        }
                    }
                }
            }
        }
    }
    false
}

// ============================================================================
// All Extended Instruction Rules
// ============================================================================

/// Returns all extended instruction validation rules.
pub fn all_ext_inst_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        &GlslFloatOpsRule,
        &GlslIntOpsRule,
        &GlslTrigOpsRule,
        &GlslPackUnpackRule,
        &GlslGeometryOpsRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glsl_opcode_names() {
        assert_eq!(get_glsl_name(glsl::SIN), "Sin");
        assert_eq!(get_glsl_name(glsl::COS), "Cos");
        assert_eq!(get_glsl_name(glsl::SQRT), "Sqrt");
        assert_eq!(get_glsl_name(glsl::CROSS), "Cross");
        assert_eq!(get_glsl_name(999), "Unknown");
    }
}
