//! OpenCL.std extended instruction validation.
//!
//! This module validates OpenCL.std extended instructions including:
//! - Floating-point math operations
//! - Integer operations
//! - Geometry operations (cross, distance, length, normalize)

use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::Op;

use crate::validation::context::{ValidationContext, ValidationRule};
use crate::validation::ValidationResult;
use crate::validation::error::ValidationError;
use crate::validation::type_ext::{DefaultTypeResolver, TypeResolver};
use crate::validation::types::{Id, ResultId};

use super::glsl::get_vector_component_count;

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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                            }.into());
                        }

                        // Check vector dimension (must be 2, 3, 4, 8, or 16)
                        let num_components =
                            get_vector_component_count(result_type, ctx.definitions).unwrap_or(1);
                        if num_components > 4 && num_components != 8 && num_components != 16 {
                            return Err(ValidationError::OpenClExtInstBadVectorDimension {
                                function: function_id,
                                block: block_id,
                            }.into());
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
                                                    }.into(),
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                            }.into());
                        }

                        // Check vector dimension
                        let num_components =
                            get_vector_component_count(result_type, ctx.definitions).unwrap_or(1);
                        if num_components > 4 && num_components != 8 && num_components != 16 {
                            return Err(ValidationError::OpenClExtInstBadVectorDimension {
                                function: function_id,
                                block: block_id,
                            }.into());
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
                                                    }.into(),
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

    fn validate(&self, ctx: &ValidationContext<'_>) -> ValidationResult {
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
                                        }.into(),
                        );
                                }
                                let num_components = num_components.unwrap_or(0);
                                if num_components != 3 && num_components != 4 {
                                    return Err(
                                        ValidationError::OpenClCrossBadVectorDimension {
                                            function: function_id,
                                            block: block_id,
                                        }.into(),
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
                                        }.into(),
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
                                        }.into(),
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_float_math_op_detection() {
        assert!(is_opencl_float_math_op(opencl::SIN));
        assert!(is_opencl_float_math_op(opencl::COS));
        assert!(is_opencl_float_math_op(opencl::EXP));
        assert!(is_opencl_float_math_op(opencl::SQRT));
        assert!(is_opencl_float_math_op(opencl::FABS));
        assert!(is_opencl_float_math_op(opencl::NATIVE_SIN));
        assert!(is_opencl_float_math_op(opencl::HALF_COS));

        assert!(!is_opencl_float_math_op(opencl::S_ABS));
        assert!(!is_opencl_float_math_op(opencl::CLZ));
    }

    #[test]
    fn test_opencl_int_op_detection() {
        assert!(is_opencl_int_op(opencl::S_ABS));
        assert!(is_opencl_int_op(opencl::U_ABS));
        assert!(is_opencl_int_op(opencl::CLZ));
        assert!(is_opencl_int_op(opencl::CTZ));
        assert!(is_opencl_int_op(opencl::POPCOUNT));

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
}
