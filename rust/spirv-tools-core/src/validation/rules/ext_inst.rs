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
// OpenCL.std Opcode Constants
// ============================================================================

/// OpenCL.std opcodes for extended instructions.
#[allow(dead_code)]
mod opencl {
    // Math functions (float, return float)
    pub const ACOS: u32 = 0;
    pub const ACOSH: u32 = 1;
    pub const ACOSPI: u32 = 2;
    pub const ASIN: u32 = 3;
    pub const ASINH: u32 = 4;
    pub const ASINPI: u32 = 5;
    pub const ATAN: u32 = 6;
    pub const ATAN2: u32 = 7;
    pub const ATANH: u32 = 8;
    pub const ATANPI: u32 = 9;
    pub const ATAN2PI: u32 = 10;
    pub const CBRT: u32 = 11;
    pub const CEIL: u32 = 12;
    pub const COPYSIGN: u32 = 13;
    pub const COS: u32 = 14;
    pub const COSH: u32 = 15;
    pub const COSPI: u32 = 16;
    pub const ERFC: u32 = 17;
    pub const ERF: u32 = 18;
    pub const EXP: u32 = 19;
    pub const EXP2: u32 = 20;
    pub const EXP10: u32 = 21;
    pub const EXPM1: u32 = 22;
    pub const FABS: u32 = 23;
    pub const FDIM: u32 = 24;
    pub const FLOOR: u32 = 25;
    pub const FMA: u32 = 26;
    pub const FMAX: u32 = 27;
    pub const FMIN: u32 = 28;
    pub const FMOD: u32 = 29;
    pub const FRACT: u32 = 30;
    pub const FREXP: u32 = 31;
    pub const HYPOT: u32 = 32;
    pub const ILOGB: u32 = 33;
    pub const LDEXP: u32 = 34;
    pub const LGAMMA: u32 = 35;
    pub const LGAMMA_R: u32 = 36;
    pub const LOG: u32 = 37;
    pub const LOG2: u32 = 38;
    pub const LOG10: u32 = 39;
    pub const LOG1P: u32 = 40;
    pub const LOGB: u32 = 41;
    pub const MAD: u32 = 42;
    pub const MAXMAG: u32 = 43;
    pub const MINMAG: u32 = 44;
    pub const MODF: u32 = 45;
    pub const NAN: u32 = 46;
    pub const NEXTAFTER: u32 = 47;
    pub const POW: u32 = 48;
    pub const POWN: u32 = 49;
    pub const POWR: u32 = 50;
    pub const REMAINDER: u32 = 51;
    pub const REMQUO: u32 = 52;
    pub const RINT: u32 = 53;
    pub const ROOTN: u32 = 54;
    pub const ROUND: u32 = 55;
    pub const RSQRT: u32 = 56;
    pub const SIN: u32 = 57;
    pub const SINCOS: u32 = 58;
    pub const SINH: u32 = 59;
    pub const SINPI: u32 = 60;
    pub const SQRT: u32 = 61;
    pub const TAN: u32 = 62;
    pub const TANH: u32 = 63;
    pub const TANPI: u32 = 64;
    pub const TGAMMA: u32 = 65;
    pub const TRUNC: u32 = 66;

    // Half-precision math (float, return float)
    pub const HALF_COS: u32 = 67;
    pub const HALF_DIVIDE: u32 = 68;
    pub const HALF_EXP: u32 = 69;
    pub const HALF_EXP2: u32 = 70;
    pub const HALF_EXP10: u32 = 71;
    pub const HALF_LOG: u32 = 72;
    pub const HALF_LOG2: u32 = 73;
    pub const HALF_LOG10: u32 = 74;
    pub const HALF_POWR: u32 = 75;
    pub const HALF_RECIP: u32 = 76;
    pub const HALF_RSQRT: u32 = 77;
    pub const HALF_SIN: u32 = 78;
    pub const HALF_SQRT: u32 = 79;
    pub const HALF_TAN: u32 = 80;

    // Native math (float, return float)
    pub const NATIVE_COS: u32 = 81;
    pub const NATIVE_DIVIDE: u32 = 82;
    pub const NATIVE_EXP: u32 = 83;
    pub const NATIVE_EXP2: u32 = 84;
    pub const NATIVE_EXP10: u32 = 85;
    pub const NATIVE_LOG: u32 = 86;
    pub const NATIVE_LOG2: u32 = 87;
    pub const NATIVE_LOG10: u32 = 88;
    pub const NATIVE_POWR: u32 = 89;
    pub const NATIVE_RECIP: u32 = 90;
    pub const NATIVE_RSQRT: u32 = 91;
    pub const NATIVE_SIN: u32 = 92;
    pub const NATIVE_SQRT: u32 = 93;
    pub const NATIVE_TAN: u32 = 94;

    // Integer functions
    pub const S_ABS: u32 = 141;
    pub const S_ABS_DIFF: u32 = 142;
    pub const S_ADD_SAT: u32 = 143;
    pub const U_ADD_SAT: u32 = 144;
    pub const S_HADD: u32 = 145;
    pub const U_HADD: u32 = 146;
    pub const S_RHADD: u32 = 147;
    pub const U_RHADD: u32 = 148;
    pub const S_CLAMP: u32 = 149;
    pub const U_CLAMP: u32 = 150;
    pub const CLZ: u32 = 151;
    pub const CTZ: u32 = 152;
    pub const S_MAD_HI: u32 = 153;
    pub const U_MAD_SAT: u32 = 154;
    pub const S_MAD_SAT: u32 = 155;
    pub const S_MAX: u32 = 156;
    pub const U_MAX: u32 = 157;
    pub const S_MIN: u32 = 158;
    pub const U_MIN: u32 = 159;
    pub const S_MUL_HI: u32 = 160;
    pub const ROTATE: u32 = 161;
    pub const S_SUB_SAT: u32 = 162;
    pub const U_SUB_SAT: u32 = 163;
    pub const U_UPSAMPLE: u32 = 164;
    pub const S_UPSAMPLE: u32 = 165;
    pub const POPCOUNT: u32 = 166;
    pub const S_MAD24: u32 = 167;
    pub const U_MAD24: u32 = 168;
    pub const S_MUL24: u32 = 169;
    pub const U_MUL24: u32 = 170;
    pub const U_ABS: u32 = 201;
    pub const U_ABS_DIFF: u32 = 202;
    pub const U_MUL_HI: u32 = 203;
    pub const U_MAD_HI: u32 = 204;

    // Common functions
    pub const FCLAMP: u32 = 95;
    pub const DEGREES: u32 = 96;
    pub const FMAX_COMMON: u32 = 97;
    pub const FMIN_COMMON: u32 = 98;
    pub const MIX: u32 = 99;
    pub const RADIANS: u32 = 100;
    pub const STEP: u32 = 101;
    pub const SMOOTHSTEP: u32 = 102;
    pub const SIGN: u32 = 103;

    // Geometric functions
    pub const CROSS: u32 = 104;
    pub const DISTANCE: u32 = 105;
    pub const LENGTH: u32 = 106;
    pub const NORMALIZE: u32 = 107;
    pub const FAST_DISTANCE: u32 = 108;
    pub const FAST_LENGTH: u32 = 109;
    pub const FAST_NORMALIZE: u32 = 110;

    // Relational functions
    pub const BITSELECT: u32 = 186;
    pub const SELECT: u32 = 187;

    // Vector load/store
    pub const VLOADN: u32 = 171;
    pub const VSTOREN: u32 = 172;
    pub const VLOAD_HALF: u32 = 173;
    pub const VLOAD_HALFN: u32 = 174;
    pub const VSTORE_HALF: u32 = 175;
    pub const VSTORE_HALF_R: u32 = 176;
    pub const VSTORE_HALFN: u32 = 177;
    pub const VSTORE_HALFN_R: u32 = 178;
    pub const VLOADA_HALFN: u32 = 179;
    pub const VSTOREA_HALFN: u32 = 180;
    pub const VSTOREA_HALFN_R: u32 = 181;

    // Shuffle
    pub const SHUFFLE: u32 = 182;
    pub const SHUFFLE2: u32 = 183;

    // Printf and prefetch
    pub const PRINTF: u32 = 184;
    pub const PREFETCH: u32 = 185;
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

/// Get struct member types.
fn get_struct_member_types(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<Vec<u32>> {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypeStruct {
                let mut member_types = Vec::new();
                for operand in &type_inst.operands {
                    if let Operand::IdRef(member_type_id) = operand {
                        member_types.push(*member_type_id);
                    }
                }
                return Some(member_types);
            }
        }
    }
    None
}

/// Get the component count of a vector type.
fn get_vector_component_count(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypeVector {
                if let Some(Operand::LiteralBit32(count)) = type_inst.operands.get(1) {
                    return Some(*count);
                }
            }
        }
    }
    None
}

/// Get the storage class of a pointer type.
fn get_pointer_storage_class(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<rspirv::spirv::StorageClass> {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypePointer {
                if let Some(Operand::StorageClass(sc)) = type_inst.operands.first() {
                    return Some(*sc);
                }
            }
        }
    }
    None
}

/// Get the pointee type of a pointer type.
fn get_pointee_type(
    type_id: u32,
    definitions: &HashMap<ResultId, Instruction>,
) -> Option<u32> {
    if let Ok(result_id) = ResultId::try_from(type_id) {
        if let Some(type_inst) = definitions.get(&result_id) {
            if type_inst.class.opcode == Op::TypePointer {
                if let Some(Operand::IdRef(pointee_type)) = type_inst.operands.get(1) {
                    return Some(*pointee_type);
                }
            }
        }
    }
    None
}

// ============================================================================
// GLSL ModfStruct/FrexpStruct Rule
// ============================================================================

/// Validates GLSL.std.450 ModfStruct and FrexpStruct operations.
///
/// ModfStruct:
/// - Result Type must be a struct with two identical float scalar/vector members
/// - Operand X must have the same type as the struct members
///
/// FrexpStruct:
/// - Result Type must be a struct with:
///   - First member: float scalar/vector
///   - Second member: 32-bit int scalar/vector (or 16-bit with extension)
///   - Same component count for both members
/// - Operand X must have the same type as the first struct member
pub struct GlslStructOpsRule;

impl ValidationRule for GlslStructOpsRule {
    fn name(&self) -> &'static str {
        "glsl-struct-ops"
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
                        glsl::MODF_STRUCT => {
                            if let Some(result_type) = inst.result_type {
                                // Result must be a struct with two identical float members
                                if let Some(member_types) =
                                    get_struct_member_types(result_type, ctx.definitions)
                                {
                                    if member_types.len() != 2 {
                                        return Err(ValidationError::GlslModfStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // Both members must be the same type
                                    if member_types[0] != member_types[1] {
                                        return Err(ValidationError::GlslModfStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // First member must be float scalar or vector
                                    if !resolver.is_float_scalar_or_vector(
                                        member_types[0],
                                        ctx.definitions,
                                    ) {
                                        return Err(ValidationError::GlslModfStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // Operand X must have the same type as the struct members
                                    if let Some(Operand::IdRef(x_id)) = inst.operands.get(2) {
                                        if let Ok(x_result_id) = ResultId::try_from(*x_id) {
                                            if let Some(x_inst) = ctx.definitions.get(&x_result_id)
                                            {
                                                if let Some(x_type) = x_inst.result_type {
                                                    if x_type != member_types[0] {
                                                        return Err(
                                                            ValidationError::GlslModfStructOperandMismatch {
                                                                function: function_id,
                                                                block: block_id,
                                                            },
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    return Err(ValidationError::GlslModfStructBadResult {
                                        function: function_id,
                                        block: block_id,
                                    });
                                }
                            }
                        }

                        glsl::FREXP_STRUCT => {
                            if let Some(result_type) = inst.result_type {
                                // Result must be a struct with two members
                                if let Some(member_types) =
                                    get_struct_member_types(result_type, ctx.definitions)
                                {
                                    if member_types.len() != 2 {
                                        return Err(ValidationError::GlslFrexpStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // First member must be float scalar or vector
                                    if !resolver.is_float_scalar_or_vector(
                                        member_types[0],
                                        ctx.definitions,
                                    ) {
                                        return Err(ValidationError::GlslFrexpStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // Second member must be 32-bit int scalar or vector
                                    if !resolver.is_int_scalar_or_vector(
                                        member_types[1],
                                        ctx.definitions,
                                    ) {
                                        return Err(ValidationError::GlslFrexpStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // Check second member is 32-bit int
                                    if let Ok(int_type_id) = ResultId::try_from(member_types[1]) {
                                        if let Some(int_type_inst) =
                                            ctx.definitions.get(&int_type_id)
                                        {
                                            let bit_width =
                                                get_int_bit_width(int_type_inst, ctx.definitions);
                                            // Allow 32-bit, or 16-bit with extension
                                            let has_amd_ext = ctx
                                                .extensions
                                                .values
                                                .iter()
                                                .any(|ext| ext.as_str() == "SPV_AMD_gpu_shader_int16");
                                            if bit_width != Some(32)
                                                && !(has_amd_ext && bit_width == Some(16))
                                            {
                                                return Err(
                                                    ValidationError::GlslFrexpStructBadResult {
                                                        function: function_id,
                                                        block: block_id,
                                                    },
                                                );
                                            }
                                        }
                                    }

                                    // Both members must have the same component count
                                    let float_components = get_vector_component_count(
                                        member_types[0],
                                        ctx.definitions,
                                    )
                                    .unwrap_or(1);
                                    let int_components = get_vector_component_count(
                                        member_types[1],
                                        ctx.definitions,
                                    )
                                    .unwrap_or(1);
                                    if float_components != int_components {
                                        return Err(ValidationError::GlslFrexpStructBadResult {
                                            function: function_id,
                                            block: block_id,
                                        });
                                    }

                                    // Operand X must have the same type as the first struct member
                                    if let Some(Operand::IdRef(x_id)) = inst.operands.get(2) {
                                        if let Ok(x_result_id) = ResultId::try_from(*x_id) {
                                            if let Some(x_inst) = ctx.definitions.get(&x_result_id)
                                            {
                                                if let Some(x_type) = x_inst.result_type {
                                                    if x_type != member_types[0] {
                                                        return Err(
                                                            ValidationError::GlslFrexpStructOperandMismatch {
                                                                function: function_id,
                                                                block: block_id,
                                                            },
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    return Err(ValidationError::GlslFrexpStructBadResult {
                                        function: function_id,
                                        block: block_id,
                                    });
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
// GLSL Ldexp Rule
// ============================================================================

/// Validates GLSL.std.450 Ldexp operation.
///
/// Ldexp(x, exp):
/// - Result Type must be float scalar or vector
/// - Operand X must have the same type as Result Type
/// - Operand Exp must be 32-bit int scalar or vector with same component count
pub struct GlslLdexpRule;

impl ValidationRule for GlslLdexpRule {
    fn name(&self) -> &'static str {
        "glsl-ldexp"
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

                    if opcode != glsl::LDEXP {
                        continue;
                    }

                    if let Some(result_type) = inst.result_type {
                        // Result Type must be float scalar or vector
                        if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::ExtInstResultTypeMustBeFloat {
                                function: function_id,
                                block: block_id,
                                ext_inst_name: "Ldexp",
                            });
                        }

                        // Operand X must have the same type as Result Type
                        if let Some(Operand::IdRef(x_id)) = inst.operands.get(2) {
                            if let Ok(x_result_id) = ResultId::try_from(*x_id) {
                                if let Some(x_inst) = ctx.definitions.get(&x_result_id) {
                                    if let Some(x_type) = x_inst.result_type {
                                        if x_type != result_type {
                                            return Err(
                                                ValidationError::ExtInstOperandTypeMismatch {
                                                    function: function_id,
                                                    block: block_id,
                                                    ext_inst_name: "Ldexp",
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Operand Exp must be 32-bit int scalar or vector
                        if let Some(Operand::IdRef(exp_id)) = inst.operands.get(3) {
                            if let Ok(exp_result_id) = ResultId::try_from(*exp_id) {
                                if let Some(exp_inst) = ctx.definitions.get(&exp_result_id) {
                                    if let Some(exp_type) = exp_inst.result_type {
                                        if !resolver
                                            .is_int_scalar_or_vector(exp_type, ctx.definitions)
                                        {
                                            return Err(ValidationError::GlslLdexpExpMustBeInt {
                                                function: function_id,
                                                block: block_id,
                                            });
                                        }

                                        // Check bit width is 32
                                        if let Ok(exp_type_id) = ResultId::try_from(exp_type) {
                                            if let Some(exp_type_inst) =
                                                ctx.definitions.get(&exp_type_id)
                                            {
                                                let bit_width = get_int_bit_width(
                                                    exp_type_inst,
                                                    ctx.definitions,
                                                );
                                                if bit_width != Some(32) {
                                                    return Err(
                                                        ValidationError::GlslLdexpExpMustBe32Bit {
                                                            function: function_id,
                                                            block: block_id,
                                                        },
                                                    );
                                                }
                                            }
                                        }

                                        // Check component count matches
                                        let result_components =
                                            get_vector_component_count(result_type, ctx.definitions)
                                                .unwrap_or(1);
                                        let exp_components =
                                            get_vector_component_count(exp_type, ctx.definitions)
                                                .unwrap_or(1);
                                        if result_components != exp_components {
                                            return Err(
                                                ValidationError::GlslLdexpComponentCountMismatch {
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
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// GLSL InterpolateAt Rule
// ============================================================================

/// Validates GLSL.std.450 InterpolateAtCentroid/Sample/Offset operations.
///
/// These operations:
/// - Require InterpolationFunction capability
/// - Result Type must be 32-bit float scalar or vector
/// - Interpolant must be a pointer to Input storage class
/// - Interpolant's pointee type must match Result Type
/// - InterpolateAtSample: Sample must be 32-bit int
/// - InterpolateAtOffset: Offset must be vec2 of 32-bit floats
pub struct GlslInterpolateRule;

impl ValidationRule for GlslInterpolateRule {
    fn name(&self) -> &'static str {
        "glsl-interpolate"
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

                    let is_interpolate = matches!(
                        opcode,
                        glsl::INTERPOLATE_AT_CENTROID
                            | glsl::INTERPOLATE_AT_SAMPLE
                            | glsl::INTERPOLATE_AT_OFFSET
                    );

                    if !is_interpolate {
                        continue;
                    }

                    // Check InterpolationFunction capability
                    if !ctx
                        .declared_capabilities
                        .contains(&rspirv::spirv::Capability::InterpolationFunction)
                    {
                        return Err(ValidationError::GlslInterpolateRequiresCapability {
                            function: function_id,
                            block: block_id,
                            ext_inst_name: get_glsl_name(opcode),
                        });
                    }

                    if let Some(result_type) = inst.result_type {
                        // Result Type must be 32-bit float scalar or vector
                        if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::ExtInstResultTypeMustBeFloat {
                                function: function_id,
                                block: block_id,
                                ext_inst_name: get_glsl_name(opcode),
                            });
                        }

                        // Check bit width is 32
                        if let Ok(type_id) = ResultId::try_from(result_type) {
                            if let Some(type_inst) = ctx.definitions.get(&type_id) {
                                let bit_width = get_float_bit_width(type_inst, ctx.definitions);
                                if bit_width != Some(32) {
                                    return Err(
                                        ValidationError::GlslInterpolateResultMustBe32BitFloat {
                                            function: function_id,
                                            block: block_id,
                                            ext_inst_name: get_glsl_name(opcode),
                                        },
                                    );
                                }
                            }
                        }

                        // Get interpolant operand (index 2)
                        if let Some(Operand::IdRef(interpolant_id)) = inst.operands.get(2) {
                            if let Ok(interpolant_result_id) = ResultId::try_from(*interpolant_id) {
                                if let Some(interpolant_inst) =
                                    ctx.definitions.get(&interpolant_result_id)
                                {
                                    if let Some(interpolant_type) = interpolant_inst.result_type {
                                        // Interpolant must be a pointer to Input storage class
                                        let storage_class = get_pointer_storage_class(
                                            interpolant_type,
                                            ctx.definitions,
                                        );
                                        if storage_class
                                            != Some(rspirv::spirv::StorageClass::Input)
                                        {
                                            return Err(
                                                ValidationError::GlslInterpolateInputStorageClass {
                                                    function: function_id,
                                                    block: block_id,
                                                    ext_inst_name: get_glsl_name(opcode),
                                                },
                                            );
                                        }

                                        // Pointee type must match result type
                                        let pointee_type =
                                            get_pointee_type(interpolant_type, ctx.definitions);
                                        if pointee_type != Some(result_type) {
                                            return Err(
                                                ValidationError::GlslInterpolateTypeMismatch {
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

                    // InterpolateAtSample: Sample must be 32-bit int
                    if opcode == glsl::INTERPOLATE_AT_SAMPLE {
                        if let Some(Operand::IdRef(sample_id)) = inst.operands.get(3) {
                            if let Ok(sample_result_id) = ResultId::try_from(*sample_id) {
                                if let Some(sample_inst) = ctx.definitions.get(&sample_result_id) {
                                    if let Some(sample_type) = sample_inst.result_type {
                                        if !resolver.is_int_scalar(sample_type, ctx.definitions) {
                                            return Err(
                                                ValidationError::GlslInterpolateSampleMustBeInt {
                                                    function: function_id,
                                                    block: block_id,
                                                },
                                            );
                                        }

                                        // Check bit width is 32
                                        if let Ok(sample_type_id) = ResultId::try_from(sample_type)
                                        {
                                            if let Some(sample_type_inst) =
                                                ctx.definitions.get(&sample_type_id)
                                            {
                                                let bit_width = get_int_bit_width(
                                                    sample_type_inst,
                                                    ctx.definitions,
                                                );
                                                if bit_width != Some(32) {
                                                    return Err(
                                                        ValidationError::GlslInterpolateSampleMustBe32Bit {
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
                        }
                    }

                    // InterpolateAtOffset: Offset must be vec2 of 32-bit floats
                    if opcode == glsl::INTERPOLATE_AT_OFFSET {
                        if let Some(Operand::IdRef(offset_id)) = inst.operands.get(3) {
                            if let Ok(offset_result_id) = ResultId::try_from(*offset_id) {
                                if let Some(offset_inst) = ctx.definitions.get(&offset_result_id) {
                                    if let Some(offset_type) = offset_inst.result_type {
                                        // Must be vec2 float
                                        if !is_float_vec2(offset_type, ctx) {
                                            return Err(
                                                ValidationError::GlslInterpolateOffsetMustBeVec2Float {
                                                    function: function_id,
                                                    block: block_id,
                                                },
                                            );
                                        }

                                        // Check bit width is 32
                                        if let Ok(offset_type_id) = ResultId::try_from(offset_type)
                                        {
                                            if let Some(offset_type_inst) =
                                                ctx.definitions.get(&offset_type_id)
                                            {
                                                let bit_width = get_float_bit_width(
                                                    offset_type_inst,
                                                    ctx.definitions,
                                                );
                                                if bit_width != Some(32) {
                                                    return Err(
                                                        ValidationError::GlslInterpolateOffsetMustBe32Bit {
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
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// OpenCL.std Helper Functions
// ============================================================================

/// Get the OpenCL.std import ID from the module.
fn get_opencl_import_id(ctx: &ValidationContext<'_>) -> Option<u32> {
    for inst in &ctx.module.ext_inst_imports {
        if inst.class.opcode == Op::ExtInstImport {
            if let Some(Operand::LiteralString(name)) = inst.operands.first() {
                if name == "OpenCL.std" || name == "OpenCL.std.100" {
                    return inst.result_id;
                }
            }
        }
    }
    None
}

/// Check if an instruction is an OpenCL.std extended instruction.
fn is_opencl_ext_inst(inst: &Instruction, opencl_import_id: u32) -> bool {
    if inst.class.opcode != Op::ExtInst {
        return false;
    }
    if let Some(Operand::IdRef(ext_set)) = inst.operands.first() {
        return *ext_set == opencl_import_id;
    }
    false
}

/// Get the OpenCL.std opcode from an OpExtInst instruction.
fn get_opencl_opcode(inst: &Instruction) -> Option<u32> {
    if let Some(Operand::LiteralExtInstInteger(opcode)) = inst.operands.get(1) {
        return Some(*opcode);
    }
    None
}

/// Get the name of an OpenCL.std instruction.
#[allow(dead_code)]
fn get_opencl_name(opcode: u32) -> &'static str {
    match opcode {
        opencl::ACOS => "acos",
        opencl::ACOSH => "acosh",
        opencl::ACOSPI => "acospi",
        opencl::ASIN => "asin",
        opencl::ASINH => "asinh",
        opencl::ASINPI => "asinpi",
        opencl::ATAN => "atan",
        opencl::ATAN2 => "atan2",
        opencl::ATANH => "atanh",
        opencl::ATANPI => "atanpi",
        opencl::ATAN2PI => "atan2pi",
        opencl::CBRT => "cbrt",
        opencl::CEIL => "ceil",
        opencl::COPYSIGN => "copysign",
        opencl::COS => "cos",
        opencl::COSH => "cosh",
        opencl::COSPI => "cospi",
        opencl::ERFC => "erfc",
        opencl::ERF => "erf",
        opencl::EXP => "exp",
        opencl::EXP2 => "exp2",
        opencl::EXP10 => "exp10",
        opencl::EXPM1 => "expm1",
        opencl::FABS => "fabs",
        opencl::FDIM => "fdim",
        opencl::FLOOR => "floor",
        opencl::FMA => "fma",
        opencl::FMAX => "fmax",
        opencl::FMIN => "fmin",
        opencl::FMOD => "fmod",
        opencl::FRACT => "fract",
        opencl::FREXP => "frexp",
        opencl::HYPOT => "hypot",
        opencl::ILOGB => "ilogb",
        opencl::LDEXP => "ldexp",
        opencl::LGAMMA => "lgamma",
        opencl::LGAMMA_R => "lgamma_r",
        opencl::LOG => "log",
        opencl::LOG2 => "log2",
        opencl::LOG10 => "log10",
        opencl::LOG1P => "log1p",
        opencl::LOGB => "logb",
        opencl::MAD => "mad",
        opencl::MAXMAG => "maxmag",
        opencl::MINMAG => "minmag",
        opencl::MODF => "modf",
        opencl::NAN => "nan",
        opencl::NEXTAFTER => "nextafter",
        opencl::POW => "pow",
        opencl::POWN => "pown",
        opencl::POWR => "powr",
        opencl::REMAINDER => "remainder",
        opencl::REMQUO => "remquo",
        opencl::RINT => "rint",
        opencl::ROOTN => "rootn",
        opencl::ROUND => "round",
        opencl::RSQRT => "rsqrt",
        opencl::SIN => "sin",
        opencl::SINCOS => "sincos",
        opencl::SINH => "sinh",
        opencl::SINPI => "sinpi",
        opencl::SQRT => "sqrt",
        opencl::TAN => "tan",
        opencl::TANH => "tanh",
        opencl::TANPI => "tanpi",
        opencl::TGAMMA => "tgamma",
        opencl::TRUNC => "trunc",
        opencl::S_ABS => "s_abs",
        opencl::S_ABS_DIFF => "s_abs_diff",
        opencl::S_ADD_SAT => "s_add_sat",
        opencl::U_ADD_SAT => "u_add_sat",
        opencl::S_HADD => "s_hadd",
        opencl::U_HADD => "u_hadd",
        opencl::S_RHADD => "s_rhadd",
        opencl::U_RHADD => "u_rhadd",
        opencl::S_CLAMP => "s_clamp",
        opencl::U_CLAMP => "u_clamp",
        opencl::CLZ => "clz",
        opencl::CTZ => "ctz",
        opencl::S_MAD_HI => "s_mad_hi",
        opencl::U_MAD_SAT => "u_mad_sat",
        opencl::S_MAD_SAT => "s_mad_sat",
        opencl::S_MAX => "s_max",
        opencl::U_MAX => "u_max",
        opencl::S_MIN => "s_min",
        opencl::U_MIN => "u_min",
        opencl::S_MUL_HI => "s_mul_hi",
        opencl::ROTATE => "rotate",
        opencl::S_SUB_SAT => "s_sub_sat",
        opencl::U_SUB_SAT => "u_sub_sat",
        opencl::U_UPSAMPLE => "u_upsample",
        opencl::S_UPSAMPLE => "s_upsample",
        opencl::POPCOUNT => "popcount",
        opencl::CROSS => "cross",
        opencl::DISTANCE => "distance",
        opencl::LENGTH => "length",
        opencl::NORMALIZE => "normalize",
        opencl::FAST_DISTANCE => "fast_distance",
        opencl::FAST_LENGTH => "fast_length",
        opencl::FAST_NORMALIZE => "fast_normalize",
        _ => "Unknown",
    }
}

/// Check if opcode is a float math operation (result must be float scalar/vector).
fn is_opencl_float_math_op(opcode: u32) -> bool {
    matches!(
        opcode,
        opencl::ACOS
            | opencl::ACOSH
            | opencl::ACOSPI
            | opencl::ASIN
            | opencl::ASINH
            | opencl::ASINPI
            | opencl::ATAN
            | opencl::ATAN2
            | opencl::ATANH
            | opencl::ATANPI
            | opencl::ATAN2PI
            | opencl::CBRT
            | opencl::CEIL
            | opencl::COPYSIGN
            | opencl::COS
            | opencl::COSH
            | opencl::COSPI
            | opencl::ERFC
            | opencl::ERF
            | opencl::EXP
            | opencl::EXP2
            | opencl::EXP10
            | opencl::EXPM1
            | opencl::FABS
            | opencl::FDIM
            | opencl::FLOOR
            | opencl::FMA
            | opencl::FMAX
            | opencl::FMIN
            | opencl::FMOD
            | opencl::HYPOT
            | opencl::LGAMMA
            | opencl::LOG
            | opencl::LOG2
            | opencl::LOG10
            | opencl::LOG1P
            | opencl::LOGB
            | opencl::MAD
            | opencl::MAXMAG
            | opencl::MINMAG
            | opencl::NEXTAFTER
            | opencl::POW
            | opencl::POWR
            | opencl::REMAINDER
            | opencl::RINT
            | opencl::ROUND
            | opencl::RSQRT
            | opencl::SIN
            | opencl::SINH
            | opencl::SINPI
            | opencl::SQRT
            | opencl::TAN
            | opencl::TANH
            | opencl::TANPI
            | opencl::TGAMMA
            | opencl::TRUNC
            | opencl::HALF_COS
            | opencl::HALF_DIVIDE
            | opencl::HALF_EXP
            | opencl::HALF_EXP2
            | opencl::HALF_EXP10
            | opencl::HALF_LOG
            | opencl::HALF_LOG2
            | opencl::HALF_LOG10
            | opencl::HALF_POWR
            | opencl::HALF_RECIP
            | opencl::HALF_RSQRT
            | opencl::HALF_SIN
            | opencl::HALF_SQRT
            | opencl::HALF_TAN
            | opencl::NATIVE_COS
            | opencl::NATIVE_DIVIDE
            | opencl::NATIVE_EXP
            | opencl::NATIVE_EXP2
            | opencl::NATIVE_EXP10
            | opencl::NATIVE_LOG
            | opencl::NATIVE_LOG2
            | opencl::NATIVE_LOG10
            | opencl::NATIVE_POWR
            | opencl::NATIVE_RECIP
            | opencl::NATIVE_RSQRT
            | opencl::NATIVE_SIN
            | opencl::NATIVE_SQRT
            | opencl::NATIVE_TAN
            | opencl::FCLAMP
            | opencl::DEGREES
            | opencl::FMAX_COMMON
            | opencl::FMIN_COMMON
            | opencl::MIX
            | opencl::RADIANS
            | opencl::STEP
            | opencl::SMOOTHSTEP
            | opencl::SIGN
    )
}

/// Check if opcode is an integer operation (result must be int scalar/vector).
fn is_opencl_int_op(opcode: u32) -> bool {
    matches!(
        opcode,
        opencl::S_ABS
            | opencl::S_ABS_DIFF
            | opencl::S_ADD_SAT
            | opencl::U_ADD_SAT
            | opencl::S_HADD
            | opencl::U_HADD
            | opencl::S_RHADD
            | opencl::U_RHADD
            | opencl::S_CLAMP
            | opencl::U_CLAMP
            | opencl::CLZ
            | opencl::CTZ
            | opencl::S_MAD_HI
            | opencl::U_MAD_SAT
            | opencl::S_MAD_SAT
            | opencl::S_MAX
            | opencl::U_MAX
            | opencl::S_MIN
            | opencl::U_MIN
            | opencl::S_MUL_HI
            | opencl::ROTATE
            | opencl::S_SUB_SAT
            | opencl::U_SUB_SAT
            | opencl::POPCOUNT
            | opencl::U_ABS
            | opencl::U_ABS_DIFF
            | opencl::U_MUL_HI
            | opencl::U_MAD_HI
    )
}

// ============================================================================
// OpenCL.std Validation Rules
// ============================================================================

/// Validates OpenCL.std floating-point math operations.
///
/// These operations require:
/// - Result Type to be a float scalar or vector
/// - Vector dimension to be 2, 3, 4, 8, or 16
/// - All operands to match Result Type
pub struct OpenClFloatOpsRule;

impl ValidationRule for OpenClFloatOpsRule {
    fn name(&self) -> &'static str {
        "opencl-float-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let opencl_import_id = match get_opencl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No OpenCL.std import
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
                    if !is_opencl_ext_inst(inst, opencl_import_id) {
                        continue;
                    }

                    let opcode = match get_opencl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    if !is_opencl_float_math_op(opcode) {
                        continue;
                    }

                    if let Some(result_type) = inst.result_type {
                        // Result Type must be float scalar or vector
                        if !resolver.is_float_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::OpenClExtInstResultTypeMustBeFloat {
                                function: function_id,
                                block: block_id,
                            });
                        }

                        // Check vector dimension (must be 2, 3, 4, 8, or 16)
                        let num_components =
                            get_vector_component_count(result_type, ctx.definitions).unwrap_or(1);
                        if num_components > 4 && num_components != 8 && num_components != 16 {
                            return Err(ValidationError::OpenClExtInstBadVectorDimension {
                                function: function_id,
                                block: block_id,
                            });
                        }

                        // All operands must match Result Type
                        for i in 2..inst.operands.len() {
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(i) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if operand_type != result_type {
                                                return Err(
                                                    ValidationError::OpenClExtInstOperandTypeMismatch {
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
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpenCL.std integer operations.
///
/// These operations require:
/// - Result Type to be an int scalar or vector
/// - Vector dimension to be 2, 3, 4, 8, or 16
/// - All operands to match Result Type
pub struct OpenClIntOpsRule;

impl ValidationRule for OpenClIntOpsRule {
    fn name(&self) -> &'static str {
        "opencl-int-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let opencl_import_id = match get_opencl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No OpenCL.std import
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
                    if !is_opencl_ext_inst(inst, opencl_import_id) {
                        continue;
                    }

                    let opcode = match get_opencl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    if !is_opencl_int_op(opcode) {
                        continue;
                    }

                    if let Some(result_type) = inst.result_type {
                        // Result Type must be int scalar or vector
                        if !resolver.is_int_scalar_or_vector(result_type, ctx.definitions) {
                            return Err(ValidationError::OpenClExtInstResultTypeMustBeInt {
                                function: function_id,
                                block: block_id,
                            });
                        }

                        // Check vector dimension
                        let num_components =
                            get_vector_component_count(result_type, ctx.definitions).unwrap_or(1);
                        if num_components > 4 && num_components != 8 && num_components != 16 {
                            return Err(ValidationError::OpenClExtInstBadVectorDimension {
                                function: function_id,
                                block: block_id,
                            });
                        }

                        // All operands must match Result Type
                        for i in 2..inst.operands.len() {
                            if let Some(Operand::IdRef(operand_id)) = inst.operands.get(i) {
                                if let Ok(operand_result_id) = ResultId::try_from(*operand_id) {
                                    if let Some(operand_inst) =
                                        ctx.definitions.get(&operand_result_id)
                                    {
                                        if let Some(operand_type) = operand_inst.result_type {
                                            if operand_type != result_type {
                                                return Err(
                                                    ValidationError::OpenClExtInstOperandTypeMismatch {
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
                    }
                }
            }
        }

        Ok(())
    }
}

/// Validates OpenCL.std geometric operations.
///
/// - cross: Result Type must be 3-component or 4-component float vector
/// - distance/length/fast_distance/fast_length: Result Type must be float scalar
/// - normalize/fast_normalize: Result Type must be float vector
pub struct OpenClGeometryOpsRule;

impl ValidationRule for OpenClGeometryOpsRule {
    fn name(&self) -> &'static str {
        "opencl-geometry-ops"
    }

    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
        let opencl_import_id = match get_opencl_import_id(ctx) {
            Some(id) => id,
            None => return Ok(()), // No OpenCL.std import
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
                    if !is_opencl_ext_inst(inst, opencl_import_id) {
                        continue;
                    }

                    let opcode = match get_opencl_opcode(inst) {
                        Some(op) => op,
                        None => continue,
                    };

                    if let Some(result_type) = inst.result_type {
                        match opcode {
                            opencl::CROSS => {
                                // Result Type must be 3 or 4 component float vector
                                // Check it's float scalar or vector and is a vector (has component count)
                                let is_float =
                                    resolver.is_float_scalar_or_vector(result_type, ctx.definitions);
                                let num_components =
                                    get_vector_component_count(result_type, ctx.definitions);
                                if !is_float || num_components.is_none() {
                                    return Err(
                                        ValidationError::OpenClCrossResultMustBeFloatVector {
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                                let num_components = num_components.unwrap_or(0);
                                if num_components != 3 && num_components != 4 {
                                    return Err(
                                        ValidationError::OpenClCrossBadVectorDimension {
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                            opencl::DISTANCE
                            | opencl::LENGTH
                            | opencl::FAST_DISTANCE
                            | opencl::FAST_LENGTH => {
                                // Result Type must be float scalar
                                if !resolver.is_float_scalar(result_type, ctx.definitions) {
                                    return Err(
                                        ValidationError::OpenClGeometryResultMustBeFloatScalar {
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                            opencl::NORMALIZE | opencl::FAST_NORMALIZE => {
                                // Result Type must be float vector (float scalar or vector, but must be vector)
                                let is_float =
                                    resolver.is_float_scalar_or_vector(result_type, ctx.definitions);
                                let is_vector =
                                    get_vector_component_count(result_type, ctx.definitions)
                                        .is_some();
                                if !is_float || !is_vector {
                                    return Err(
                                        ValidationError::OpenClNormalizeResultMustBeFloatVector {
                                            function: function_id,
                                            block: block_id,
                                        },
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(())
    }
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
        &GlslStructOpsRule,
        &GlslLdexpRule,
        &GlslInterpolateRule,
        // OpenCL.std rules
        &OpenClFloatOpsRule,
        &OpenClIntOpsRule,
        &OpenClGeometryOpsRule,
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

    #[test]
    fn test_glsl_struct_ops_names() {
        // Test that we recognize ModfStruct and FrexpStruct
        assert_eq!(get_glsl_name(glsl::MODF_STRUCT), "ModfStruct");
        assert_eq!(get_glsl_name(glsl::FREXP_STRUCT), "FrexpStruct");
    }

    #[test]
    fn test_glsl_ldexp_name() {
        assert_eq!(get_glsl_name(glsl::LDEXP), "Ldexp");
    }

    #[test]
    fn test_glsl_interpolate_names() {
        assert_eq!(get_glsl_name(glsl::INTERPOLATE_AT_CENTROID), "InterpolateAtCentroid");
        assert_eq!(get_glsl_name(glsl::INTERPOLATE_AT_SAMPLE), "InterpolateAtSample");
        assert_eq!(get_glsl_name(glsl::INTERPOLATE_AT_OFFSET), "InterpolateAtOffset");
    }

    #[test]
    fn test_all_ext_inst_rules_includes_new_rules() {
        let rules = all_ext_inst_rules();
        let names: Vec<&str> = rules.iter().map(|r| r.name()).collect();

        // GLSL rules
        assert!(names.contains(&"glsl-struct-ops"), "Missing glsl-struct-ops rule");
        assert!(names.contains(&"glsl-ldexp"), "Missing glsl-ldexp rule");
        assert!(names.contains(&"glsl-interpolate"), "Missing glsl-interpolate rule");

        // OpenCL rules
        assert!(names.contains(&"opencl-float-ops"), "Missing opencl-float-ops rule");
        assert!(names.contains(&"opencl-int-ops"), "Missing opencl-int-ops rule");
        assert!(names.contains(&"opencl-geometry-ops"), "Missing opencl-geometry-ops rule");
    }

    #[test]
    fn test_opencl_float_math_op_detection() {
        // Test that we correctly identify float math ops
        assert!(is_opencl_float_math_op(opencl::SIN));
        assert!(is_opencl_float_math_op(opencl::COS));
        assert!(is_opencl_float_math_op(opencl::EXP));
        assert!(is_opencl_float_math_op(opencl::SQRT));
        assert!(is_opencl_float_math_op(opencl::FABS));
        assert!(is_opencl_float_math_op(opencl::NATIVE_SIN));
        assert!(is_opencl_float_math_op(opencl::HALF_COS));

        // Integer ops should not be float ops
        assert!(!is_opencl_float_math_op(opencl::S_ABS));
        assert!(!is_opencl_float_math_op(opencl::CLZ));
    }

    #[test]
    fn test_opencl_int_op_detection() {
        // Test that we correctly identify int ops
        assert!(is_opencl_int_op(opencl::S_ABS));
        assert!(is_opencl_int_op(opencl::U_ABS));
        assert!(is_opencl_int_op(opencl::CLZ));
        assert!(is_opencl_int_op(opencl::CTZ));
        assert!(is_opencl_int_op(opencl::POPCOUNT));

        // Float ops should not be int ops
        assert!(!is_opencl_int_op(opencl::SIN));
        assert!(!is_opencl_int_op(opencl::SQRT));
    }

    #[test]
    fn test_opencl_name_lookup() {
        assert_eq!(get_opencl_name(opencl::SIN), "sin");
        assert_eq!(get_opencl_name(opencl::COS), "cos");
        assert_eq!(get_opencl_name(opencl::SQRT), "sqrt");
        assert_eq!(get_opencl_name(opencl::S_ABS), "s_abs");
        assert_eq!(get_opencl_name(opencl::CROSS), "cross");
        assert_eq!(get_opencl_name(opencl::LENGTH), "length");
        assert_eq!(get_opencl_name(opencl::NORMALIZE), "normalize");
        assert_eq!(get_opencl_name(999), "Unknown");
    }

    #[test]
    fn test_get_int_bit_width_scalar() {
        use rspirv::dr::{Instruction, Operand};
        use std::collections::HashMap;

        // Create OpTypeInt 32 1
        let mut int_inst = Instruction::new(rspirv::spirv::Op::TypeInt, None, Some(5), vec![]);
        int_inst.operands.push(Operand::LiteralBit32(32)); // width
        int_inst.operands.push(Operand::LiteralBit32(1)); // signedness

        let definitions: HashMap<ResultId, Instruction> = HashMap::new();
        let bit_width = get_int_bit_width(&int_inst, &definitions);
        assert_eq!(bit_width, Some(32));
    }

    #[test]
    fn test_get_int_bit_width_16() {
        use rspirv::dr::{Instruction, Operand};
        use std::collections::HashMap;

        // Create OpTypeInt 16 0
        let mut int_inst = Instruction::new(rspirv::spirv::Op::TypeInt, None, Some(5), vec![]);
        int_inst.operands.push(Operand::LiteralBit32(16)); // width
        int_inst.operands.push(Operand::LiteralBit32(0)); // signedness

        let definitions: HashMap<ResultId, Instruction> = HashMap::new();
        let bit_width = get_int_bit_width(&int_inst, &definitions);
        assert_eq!(bit_width, Some(16));
    }

    #[test]
    fn test_get_float_bit_width_scalar() {
        use rspirv::dr::{Instruction, Operand};
        use std::collections::HashMap;

        // Create OpTypeFloat 32
        let mut float_inst = Instruction::new(rspirv::spirv::Op::TypeFloat, None, Some(5), vec![]);
        float_inst.operands.push(Operand::LiteralBit32(32)); // width

        let definitions: HashMap<ResultId, Instruction> = HashMap::new();
        let bit_width = get_float_bit_width(&float_inst, &definitions);
        assert_eq!(bit_width, Some(32));
    }

    #[test]
    fn test_get_float_bit_width_64() {
        use rspirv::dr::{Instruction, Operand};
        use std::collections::HashMap;

        // Create OpTypeFloat 64
        let mut float_inst = Instruction::new(rspirv::spirv::Op::TypeFloat, None, Some(5), vec![]);
        float_inst.operands.push(Operand::LiteralBit32(64)); // width

        let definitions: HashMap<ResultId, Instruction> = HashMap::new();
        let bit_width = get_float_bit_width(&float_inst, &definitions);
        assert_eq!(bit_width, Some(64));
    }

    #[test]
    fn test_glsl_pack_unpack_names() {
        assert_eq!(get_glsl_name(glsl::PACK_SNORM4X8), "PackSnorm4x8");
        assert_eq!(get_glsl_name(glsl::UNPACK_SNORM4X8), "UnpackSnorm4x8");
        assert_eq!(get_glsl_name(glsl::PACK_HALF2X16), "PackHalf2x16");
        assert_eq!(get_glsl_name(glsl::UNPACK_HALF2X16), "UnpackHalf2x16");
    }

    #[test]
    fn test_glsl_geometry_names() {
        assert_eq!(get_glsl_name(glsl::LENGTH), "Length");
        assert_eq!(get_glsl_name(glsl::DISTANCE), "Distance");
        assert_eq!(get_glsl_name(glsl::CROSS), "Cross");
        assert_eq!(get_glsl_name(glsl::NORMALIZE), "Normalize");
        assert_eq!(get_glsl_name(glsl::FACE_FORWARD), "FaceForward");
        assert_eq!(get_glsl_name(glsl::REFLECT), "Reflect");
        assert_eq!(get_glsl_name(glsl::REFRACT), "Refract");
    }

    #[test]
    fn test_glsl_exponential_names() {
        assert_eq!(get_glsl_name(glsl::EXP), "Exp");
        assert_eq!(get_glsl_name(glsl::LOG), "Log");
        assert_eq!(get_glsl_name(glsl::EXP2), "Exp2");
        assert_eq!(get_glsl_name(glsl::LOG2), "Log2");
        assert_eq!(get_glsl_name(glsl::POW), "Pow");
    }

    #[test]
    fn test_glsl_common_names() {
        assert_eq!(get_glsl_name(glsl::FABS), "FAbs");
        assert_eq!(get_glsl_name(glsl::SABS), "SAbs");
        assert_eq!(get_glsl_name(glsl::FSIGN), "FSign");
        assert_eq!(get_glsl_name(glsl::SSIGN), "SSign");
        assert_eq!(get_glsl_name(glsl::FLOOR), "Floor");
        assert_eq!(get_glsl_name(glsl::CEIL), "Ceil");
        assert_eq!(get_glsl_name(glsl::FRACT), "Fract");
    }
}
