//! E-graph driven optimizer using egglog for SPIR-V.
//!
//! This module provides the egglog-based optimizer implementation using RVSDG
//! (Regionalized Value State Dependence Graph) for control flow representation.
//!
//! Key RVSDG concepts:
//! - **Gamma nodes**: Conditional selection (replaces if-then-else + phi)
//! - **Theta nodes**: Tail-controlled loops (replaces loops + phi)
//! - **Values flow through nodes**, not through block labels
//!
//! This module contains:
//! - The egglog program with RVSDG datatypes and rewrite rules
//! - Custom primitives for operations that can't be expressed as simple rewrites
//! - The `create_spirv_egraph()` function to create a configured e-graph
//!
//! Rules are organized into separate files in the `rules/` directory:
//! - `datatypes.egg`: Core datatype definitions (Expr, Effect, ExprList)
//! - `rvsdg.egg`: RVSDG-specific rules (Gamma, Theta, Effect rewrites)
//! - `arithmetic.egg`: Integer arithmetic optimizations
//! - `bitwise.egg`: Bitwise operation optimizations
//! - `comparison.egg`: Comparison operation optimizations
//! - `logical.egg`: Logical and select/if optimizations
//! - `floating_point.egg`: Floating-point optimizations
//! - `vector.egg`: Vector construction/extraction and arithmetic
//! - `matrix.egg`: Matrix operation optimizations
//! - `glsl.egg`: GLSL extended instruction optimizations
//! - `type_conversion.egg`: Type conversion and pack/unpack optimizations
//! - `constant_folding.egg`: Constant folding rules
//! - `primitives.egg`: Rules using custom Rust primitives

use egglog::{add_primitive, EGraph};

/// The egglog program defining the SPIR-V RVSDG language and rewrite rules.
/// This is assembled from multiple rule files at compile time.
const SPIRV_EGGLOG_PROGRAM: &str = concat!(
    include_str!("rules/datatypes.egg"),
    "\n",
    include_str!("rules/rvsdg.egg"),
    "\n",
    include_str!("rules/arithmetic.egg"),
    "\n",
    include_str!("rules/bitwise.egg"),
    "\n",
    include_str!("rules/comparison.egg"),
    "\n",
    include_str!("rules/logical.egg"),
    "\n",
    include_str!("rules/floating_point.egg"),
    "\n",
    include_str!("rules/vector.egg"),
    "\n",
    include_str!("rules/matrix.egg"),
    "\n",
    include_str!("rules/glsl.egg"),
    "\n",
    include_str!("rules/type_conversion.egg"),
    "\n",
    include_str!("rules/constant_folding.egg"),
    "\n",
    include_str!("rules/memory.egg"),
    "\n",
    include_str!("rules/inlining.egg"),
    "\n",
    include_str!("rules/loop_unroll.egg"),
    "\n",
    include_str!("rules/spec_constant.egg"),
    "\n",
    include_str!("rules/sroa.egg"),
    "\n",
    include_str!("rules/advanced_loops.egg"),
    "\n",
    include_str!("rules/copy_propagation.egg"),
    "\n",
    include_str!("rules/graphics.egg"),
    "\n",
    include_str!("rules/float_conversion.egg"),
    "\n",
    include_str!("rules/cleanup.egg"),
    "\n",
    include_str!("rules/subgroup.egg"),
);

/// Rules that use custom primitives (must be loaded after primitives are registered).
const SPIRV_EGGLOG_PRIMITIVES_PROGRAM: &str = include_str!("rules/primitives.egg");

/// Error type for egglog optimization.
#[derive(Debug, thiserror::Error)]
pub enum EgglogOptError {
    #[error("egglog parse error: {0}")]
    ParseError(String),
    #[error("egglog execution error: {0}")]
    ExecutionError(String),
    #[error("extraction error: {0}")]
    ExtractionError(String),
}

// =============================================================================
// Custom Primitives
// =============================================================================

/// Reverse the bits of a 32-bit value.
fn bitreverse32(x: i64) -> i64 {
    (x as u32).reverse_bits() as i64
}

/// Reverse the bits of a 64-bit value.
fn bitreverse64(x: i64) -> i64 {
    (x as u64).reverse_bits() as i64
}

/// Check if two bitmasks are disjoint (no overlapping bits).
fn bits_disjoint(a: i64, b: i64) -> Option<()> {
    if (a & b) == 0 { Some(()) } else { None }
}

/// Check if left shift would clear all bits in a mask.
fn shl_clears_mask(mask: i64, shift: i64) -> Option<()> {
    let shift = shift as u32;
    if shift >= 64 {
        return Some(());
    }
    let surviving_mask = !((1i64 << shift) - 1);
    if (mask & surviving_mask) == 0 { Some(()) } else { None }
}

/// Check if right shift would clear all bits in a mask.
fn shr_clears_mask(mask: i64, shift: i64) -> Option<()> {
    let shift = shift as u32;

    // 32-bit semantics
    if shift >= 32 {
        return Some(());
    }
    let surviving_bits = 32u32.saturating_sub(shift);
    let surviving_mask = if surviving_bits >= 32 {
        u32::MAX
    } else {
        (1u32 << surviving_bits) - 1
    };
    let mask32 = mask as u32;
    if (mask32 & surviving_mask) == 0 {
        return Some(());
    }

    // 64-bit semantics
    if shift >= 64 {
        return Some(());
    }
    let surviving_bits = 64u32.saturating_sub(shift);
    let surviving_mask: i64 = if surviving_bits >= 64 {
        -1
    } else {
        ((1u64 << surviving_bits) - 1) as i64
    };
    if (mask & surviving_mask) == 0 { Some(()) } else { None }
}

/// Check if mask a is a superset of mask b.
fn mask_superset(a: i64, b: i64) -> Option<()> {
    if (a & b) == b { Some(()) } else { None }
}

/// Find the least significant bit position (0-indexed), or -1 if zero.
fn find_lsb(x: i64) -> i64 {
    if x == 0 {
        -1
    } else {
        (x as u64).trailing_zeros() as i64
    }
}

/// Find the most significant bit position for unsigned (0-indexed), or -1 if zero.
fn find_msb_unsigned(x: i64) -> i64 {
    if x == 0 {
        -1
    } else {
        63 - (x as u64).leading_zeros() as i64
    }
}

/// Find the most significant bit position for signed values.
/// For positive: position of highest 1-bit
/// For negative: position of highest 0-bit
fn find_msb_signed(x: i64) -> i64 {
    if x == 0 || x == -1 {
        -1
    } else if x > 0 {
        63 - (x as u64).leading_zeros() as i64
    } else {
        // For negative, find position of highest 0-bit
        63 - (!x as u64).leading_zeros() as i64
    }
}

/// Count the number of 1-bits in the value.
fn popcount(x: i64) -> i64 {
    (x as u64).count_ones() as i64
}

/// Check if a value is a power of 2 (and positive).
/// Returns Some(()) if x is a power of 2, None otherwise.
fn is_pow2(x: i64) -> Option<()> {
    if x > 0 && (x & (x - 1)) == 0 {
        Some(())
    } else {
        None
    }
}

/// Compute log base 2 of a power of 2.
/// Assumes x is a positive power of 2 (caller should guard with is-pow2).
fn log2_pow2(x: i64) -> i64 {
    if x <= 0 {
        return -1;
    }
    (x as u64).trailing_zeros() as i64
}

/// Check if an integer, when interpreted as a float, has an exact reciprocal.
/// Returns Some(()) if the float constant has an exact reciprocal representation.
fn has_exact_recip(x: i64) -> Option<()> {
    // Reinterpret the i64 as bits of a float
    // For this to work, we need to check if the value is a power of 2 float
    let f = f64::from_bits(x as u64);
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    let recip = 1.0 / f;
    if !recip.is_finite() {
        return None;
    }
    // Check if the reciprocal can be represented exactly
    // This happens for powers of 2 and some special values
    let roundtrip = 1.0 / recip;
    if roundtrip == f {
        Some(())
    } else {
        None
    }
}

/// Compute the reciprocal of a float constant (stored as i64 bits).
fn float_recip(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    let recip = 1.0 / f;
    recip.to_bits() as i64
}

// =============================================================================
// Float Arithmetic Primitives (for constant folding in FP rules)
// =============================================================================
// These primitives interpret i64 as f32 bit patterns (since SPIRV uses 32-bit floats).
// For 32-bit floats, the bit pattern is stored in the lower 32 bits of i64.

/// Multiply two f32 constants (stored as i64 bit patterns).
fn float_mul32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa * fb;
    result.to_bits() as i64
}

/// Divide two f32 constants (stored as i64 bit patterns).
fn float_div32(a: i64, b: i64) -> Option<i64> {
    let fb = f32::from_bits(b as u32);
    if fb == 0.0 {
        return None;
    }
    let fa = f32::from_bits(a as u32);
    let result = fa / fb;
    Some(result.to_bits() as i64)
}

/// Add two f32 constants (stored as i64 bit patterns).
fn float_add32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa + fb;
    result.to_bits() as i64
}

/// Subtract two f32 constants (stored as i64 bit patterns).
fn float_sub32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa - fb;
    result.to_bits() as i64
}

/// Negate an f32 constant (stored as i64 bit pattern).
fn float_neg32(a: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let result = -fa;
    result.to_bits() as i64
}

/// Check if an i64 represents f32 1.0 bit pattern.
fn is_float_one32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 1.0 { Some(()) } else { None }
}

/// Check if an i64 represents f32 0.0 bit pattern.
fn is_float_zero32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.0 { Some(()) } else { None }
}

/// Check if an i64 represents f32 -1.0 bit pattern.
fn is_float_neg_one32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -1.0 { Some(()) } else { None }
}

/// Check if an i64 represents f32 2.0 bit pattern.
fn is_float_two32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 2.0 { Some(()) } else { None }
}

/// Check if an i64 represents f32 0.5 bit pattern (for sqrt optimizations).
fn is_float_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.5 { Some(()) } else { None }
}

/// Check if an i64 represents f32 -0.5 bit pattern (for inverse sqrt optimizations).
fn is_float_neg_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -0.5 { Some(()) } else { None }
}

/// Check if an f32 constant (stored as i64 bit pattern) has an exact reciprocal.
fn has_exact_recip32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    let recip = 1.0 / f;
    if !recip.is_finite() {
        return None;
    }
    let roundtrip = 1.0 / recip;
    if roundtrip == f { Some(()) } else { None }
}

/// Compute the reciprocal of an f32 constant (stored as i64 bit pattern).
fn float_recip32(x: i64) -> i64 {
    let f = f32::from_bits(x as u32);
    let recip = 1.0f32 / f;
    recip.to_bits() as i64
}

// =============================================================================
// GLSL Transcendental Constant Folding Primitives
// =============================================================================
// These primitives evaluate GLSL math functions on float constants.
// Constants are stored as i64 (bit representation of f64).

/// Compute sin of a float constant.
fn float_sin(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sin().to_bits() as i64
}

/// Compute cos of a float constant.
fn float_cos(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.cos().to_bits() as i64
}

/// Compute tan of a float constant.
fn float_tan(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.tan().to_bits() as i64
}

/// Compute asin of a float constant.
fn float_asin(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.asin().to_bits() as i64
}

/// Compute acos of a float constant.
fn float_acos(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.acos().to_bits() as i64
}

/// Compute atan of a float constant.
fn float_atan(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.atan().to_bits() as i64
}

/// Compute atan2 of two float constants.
fn float_atan2(y: i64, x: i64) -> i64 {
    let fy = f64::from_bits(y as u64);
    let fx = f64::from_bits(x as u64);
    fy.atan2(fx).to_bits() as i64
}

/// Compute sinh of a float constant.
fn float_sinh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sinh().to_bits() as i64
}

/// Compute cosh of a float constant.
fn float_cosh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.cosh().to_bits() as i64
}

/// Compute tanh of a float constant.
fn float_tanh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.tanh().to_bits() as i64
}

/// Compute asinh of a float constant.
fn float_asinh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.asinh().to_bits() as i64
}

/// Compute acosh of a float constant.
fn float_acosh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.acosh().to_bits() as i64
}

/// Compute atanh of a float constant.
fn float_atanh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.atanh().to_bits() as i64
}

/// Compute exp of a float constant.
fn float_exp(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.exp().to_bits() as i64
}

/// Compute exp2 of a float constant.
fn float_exp2(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.exp2().to_bits() as i64
}

/// Compute log (natural logarithm) of a float constant.
fn float_log(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.ln().to_bits() as i64
}

/// Compute log2 of a float constant.
fn float_log2(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.log2().to_bits() as i64
}

/// Compute sqrt of a float constant.
fn float_sqrt(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sqrt().to_bits() as i64
}

/// Compute inverse sqrt of a float constant.
fn float_inversesqrt(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    (1.0 / f.sqrt()).to_bits() as i64
}

/// Compute pow of two float constants.
fn float_pow(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.powf(fy).to_bits() as i64
}

/// Compute floor of a float constant.
fn float_floor(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.floor().to_bits() as i64
}

/// Compute ceil of a float constant.
fn float_ceil(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.ceil().to_bits() as i64
}

/// Compute round of a float constant.
fn float_round(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.round().to_bits() as i64
}

/// Compute trunc of a float constant.
fn float_trunc(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.trunc().to_bits() as i64
}

/// Compute abs of a float constant.
fn float_abs(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.abs().to_bits() as i64
}

/// Compute sign of a float constant (-1, 0, or 1).
fn float_sign(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    if f > 0.0 {
        1.0_f64.to_bits() as i64
    } else if f < 0.0 {
        (-1.0_f64).to_bits() as i64
    } else {
        0.0_f64.to_bits() as i64
    }
}

/// Compute fract (fractional part) of a float constant.
fn float_fract(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    (f - f.floor()).to_bits() as i64
}

/// Compute min of two float constants.
fn float_min(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.min(fy).to_bits() as i64
}

/// Compute max of two float constants.
fn float_max(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.max(fy).to_bits() as i64
}

/// Compute clamp of a float constant.
fn float_clamp(x: i64, lo: i64, hi: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let flo = f64::from_bits(lo as u64);
    let fhi = f64::from_bits(hi as u64);
    fx.clamp(flo, fhi).to_bits() as i64
}

/// Compute mix (linear interpolation) of float constants.
fn float_mix(x: i64, y: i64, a: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    let fa = f64::from_bits(a as u64);
    (fx * (1.0 - fa) + fy * fa).to_bits() as i64
}

/// Compute step function.
fn float_step(edge: i64, x: i64) -> i64 {
    let fe = f64::from_bits(edge as u64);
    let fx = f64::from_bits(x as u64);
    if fx < fe { 0.0_f64 } else { 1.0_f64 }.to_bits() as i64
}

/// Compute smoothstep function.
fn float_smoothstep(edge0: i64, edge1: i64, x: i64) -> i64 {
    let e0 = f64::from_bits(edge0 as u64);
    let e1 = f64::from_bits(edge1 as u64);
    let fx = f64::from_bits(x as u64);
    let t = ((fx - e0) / (e1 - e0)).clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)).to_bits() as i64
}

// =============================================================================
// Public API
// =============================================================================

/// Create a new egglog EGraph with the SPIR-V language and rules loaded.
///
/// This creates a configured e-graph ready for SPIR-V optimization. Use this
/// with `direct::optimize_module_direct()` for whole-module optimization.
pub fn create_spirv_egraph() -> Result<EGraph, EgglogOptError> {
    let mut egraph = EGraph::default();

    // Register all custom primitives FIRST (before loading rules that use them)

    // Bit reversal primitives
    add_primitive!(&mut egraph, "bitrev32" = |a: i64| -> i64 {
        bitreverse32(a)
    });
    add_primitive!(&mut egraph, "bitrev64" = |a: i64| -> i64 {
        bitreverse64(a)
    });

    // Bit mask check primitives
    add_primitive!(&mut egraph, "bits-disjoint" = |a: i64, b: i64| -?> () {
        bits_disjoint(a, b)
    });
    add_primitive!(&mut egraph, "shl-clears-mask" = |mask: i64, shift: i64| -?> () {
        shl_clears_mask(mask, shift)
    });
    add_primitive!(&mut egraph, "shr-clears-mask" = |mask: i64, shift: i64| -?> () {
        shr_clears_mask(mask, shift)
    });
    add_primitive!(&mut egraph, "mask-superset" = |a: i64, b: i64| -?> () {
        mask_superset(a, b)
    });

    // Bit manipulation primitives
    add_primitive!(&mut egraph, "find-lsb" = |a: i64| -> i64 {
        find_lsb(a)
    });
    add_primitive!(&mut egraph, "find-msb-unsigned" = |a: i64| -> i64 {
        find_msb_unsigned(a)
    });
    add_primitive!(&mut egraph, "find-msb-signed" = |a: i64| -> i64 {
        find_msb_signed(a)
    });
    add_primitive!(&mut egraph, "popcount" = |a: i64| -> i64 {
        popcount(a)
    });

    // Guard primitive that succeeds only if a*b would not overflow.
    // Used in rule conditions to prevent rules from matching when multiplication would overflow.
    add_primitive!(&mut egraph, "mul-safe" = |a: i64, b: i64| -?> () {
        // Check if a * b would overflow
        if a.checked_mul(b).is_some() {
            Some(())
        } else {
            None
        }
    });

    // Power of 2 primitives for strength reduction (mul -> shift)
    add_primitive!(&mut egraph, "is-pow2" = |a: i64| -?> () {
        is_pow2(a)
    });
    add_primitive!(&mut egraph, "log2-pow2" = |a: i64| -> i64 {
        log2_pow2(a)
    });

    // Float reciprocal primitives for x/c -> x*(1/c) optimization (f64 versions, legacy)
    add_primitive!(&mut egraph, "has-exact-recip" = |a: i64| -?> () {
        has_exact_recip(a)
    });
    add_primitive!(&mut egraph, "float-recip" = |a: i64| -> i64 {
        float_recip(a)
    });

    // f32 arithmetic primitives for FP constant folding
    // These interpret i64 as f32 bit patterns (lower 32 bits)
    add_primitive!(&mut egraph, "float-mul32" = |a: i64, b: i64| -> i64 { float_mul32(a, b) });
    add_primitive!(&mut egraph, "float-div32" = |a: i64, b: i64| -?> i64 { float_div32(a, b) });
    add_primitive!(&mut egraph, "float-add32" = |a: i64, b: i64| -> i64 { float_add32(a, b) });
    add_primitive!(&mut egraph, "float-sub32" = |a: i64, b: i64| -> i64 { float_sub32(a, b) });
    add_primitive!(&mut egraph, "float-neg32" = |a: i64| -> i64 { float_neg32(a) });

    // f32 identity check primitives
    add_primitive!(&mut egraph, "is-float-one32" = |a: i64| -?> () { is_float_one32(a) });
    add_primitive!(&mut egraph, "is-float-zero32" = |a: i64| -?> () { is_float_zero32(a) });
    add_primitive!(&mut egraph, "is-float-neg-one32" = |a: i64| -?> () { is_float_neg_one32(a) });
    add_primitive!(&mut egraph, "is-float-two32" = |a: i64| -?> () { is_float_two32(a) });
    add_primitive!(&mut egraph, "is-float-half32" = |a: i64| -?> () { is_float_half32(a) });
    add_primitive!(&mut egraph, "is-float-neg-half32" = |a: i64| -?> () { is_float_neg_half32(a) });

    // f32 reciprocal primitives
    add_primitive!(&mut egraph, "has-exact-recip32" = |a: i64| -?> () { has_exact_recip32(a) });
    add_primitive!(&mut egraph, "float-recip32" = |a: i64| -> i64 { float_recip32(a) });

    // GLSL transcendental constant folding primitives
    add_primitive!(&mut egraph, "float-sin" = |a: i64| -> i64 { float_sin(a) });
    add_primitive!(&mut egraph, "float-cos" = |a: i64| -> i64 { float_cos(a) });
    add_primitive!(&mut egraph, "float-tan" = |a: i64| -> i64 { float_tan(a) });
    add_primitive!(&mut egraph, "float-asin" = |a: i64| -> i64 { float_asin(a) });
    add_primitive!(&mut egraph, "float-acos" = |a: i64| -> i64 { float_acos(a) });
    add_primitive!(&mut egraph, "float-atan" = |a: i64| -> i64 { float_atan(a) });
    add_primitive!(&mut egraph, "float-atan2" = |y: i64, x: i64| -> i64 { float_atan2(y, x) });
    add_primitive!(&mut egraph, "float-sinh" = |a: i64| -> i64 { float_sinh(a) });
    add_primitive!(&mut egraph, "float-cosh" = |a: i64| -> i64 { float_cosh(a) });
    add_primitive!(&mut egraph, "float-tanh" = |a: i64| -> i64 { float_tanh(a) });
    add_primitive!(&mut egraph, "float-asinh" = |a: i64| -> i64 { float_asinh(a) });
    add_primitive!(&mut egraph, "float-acosh" = |a: i64| -> i64 { float_acosh(a) });
    add_primitive!(&mut egraph, "float-atanh" = |a: i64| -> i64 { float_atanh(a) });
    add_primitive!(&mut egraph, "float-exp" = |a: i64| -> i64 { float_exp(a) });
    add_primitive!(&mut egraph, "float-exp2" = |a: i64| -> i64 { float_exp2(a) });
    add_primitive!(&mut egraph, "float-log" = |a: i64| -> i64 { float_log(a) });
    add_primitive!(&mut egraph, "float-log2" = |a: i64| -> i64 { float_log2(a) });
    add_primitive!(&mut egraph, "float-sqrt" = |a: i64| -> i64 { float_sqrt(a) });
    add_primitive!(&mut egraph, "float-inversesqrt" = |a: i64| -> i64 { float_inversesqrt(a) });
    add_primitive!(&mut egraph, "float-pow" = |x: i64, y: i64| -> i64 { float_pow(x, y) });
    add_primitive!(&mut egraph, "float-floor" = |a: i64| -> i64 { float_floor(a) });
    add_primitive!(&mut egraph, "float-ceil" = |a: i64| -> i64 { float_ceil(a) });
    add_primitive!(&mut egraph, "float-round" = |a: i64| -> i64 { float_round(a) });
    add_primitive!(&mut egraph, "float-trunc" = |a: i64| -> i64 { float_trunc(a) });
    add_primitive!(&mut egraph, "float-abs" = |a: i64| -> i64 { float_abs(a) });
    add_primitive!(&mut egraph, "float-sign" = |a: i64| -> i64 { float_sign(a) });
    add_primitive!(&mut egraph, "float-fract" = |a: i64| -> i64 { float_fract(a) });
    add_primitive!(&mut egraph, "float-min" = |x: i64, y: i64| -> i64 { float_min(x, y) });
    add_primitive!(&mut egraph, "float-max" = |x: i64, y: i64| -> i64 { float_max(x, y) });
    add_primitive!(&mut egraph, "float-clamp" = |x: i64, lo: i64, hi: i64| -> i64 { float_clamp(x, lo, hi) });
    add_primitive!(&mut egraph, "float-mix" = |x: i64, y: i64, a: i64| -> i64 { float_mix(x, y, a) });
    add_primitive!(&mut egraph, "float-step" = |edge: i64, x: i64| -> i64 { float_step(edge, x) });
    add_primitive!(&mut egraph, "float-smoothstep" = |e0: i64, e1: i64, x: i64| -> i64 { float_smoothstep(e0, e1, x) });

    // Now load the base SPIR-V language and rules (which use the primitives above)
    egraph
        .parse_and_run_program(None, SPIRV_EGGLOG_PROGRAM)
        .map_err(|e| EgglogOptError::ParseError(e.to_string()))?;

    // Load additional rules that use custom primitives
    egraph
        .parse_and_run_program(None, SPIRV_EGGLOG_PRIMITIVES_PROGRAM)
        .map_err(|e| EgglogOptError::ParseError(e.to_string()))?;

    Ok(egraph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_egraph() {
        let result = create_spirv_egraph();
        assert!(result.is_ok(), "Failed to create egraph: {:?}", result.err());
    }

    #[test]
    fn test_add_zero_optimization() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Add expression: x + 0 and (Sym "x") - they should be equivalent
        egraph.parse_and_run_program(None, r#"(let add_form (Add (Sym "x") (Const 0)))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let x_form (Sym "x"))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        // Check that they are equivalent in the e-graph
        let check = egraph.parse_and_run_program(None, "(check (= add_form x_form))");
        assert!(check.is_ok(), "Expected x + 0 to be equivalent to x in the e-graph");
    }

    #[test]
    fn test_absorption() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Add expression: x | (x & y) - should simplify to x
        egraph.parse_and_run_program(None, r#"(let root (BitOr (Sym "x") (BitAnd (Sym "x") (Sym "y"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        // Should simplify to just (Sym "x")
        assert!(result.contains("Sym") && result.contains("x") && !result.contains("BitOr"),
                "Expected absorption to (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_factoring() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Add expression: (x * 2) + (x * 3) should simplify to x * 5
        egraph.parse_and_run_program(None, r#"(let root (Add (Mul (Sym "x") (Const 2)) (Mul (Sym "x") (Const 3))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 20 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Factoring result: {}", result);
        // Should factor to (Mul (Sym "x") (Const 5))
        assert!(result.contains("Mul") && result.contains("5"),
                "Expected factoring to (Mul x 5), got: {}", result);
    }

    #[test]
    fn test_find_lsb_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FindILsb(12) = 2 (binary: 1100, lowest set bit is at position 2)
        egraph.parse_and_run_program(None, "(let root (FindILsb (Const 12)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 2"), "Expected (Const 2), got: {}", result);
    }

    #[test]
    fn test_find_lsb_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FindILsb(0) = -1
        egraph.parse_and_run_program(None, "(let root (FindILsb (Const 0)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        assert!(result.contains("Const -1"), "Expected (Const -1), got: {}", result);
    }

    #[test]
    fn test_find_msb_unsigned() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FindUMsb(8) = 3 (binary: 1000, highest set bit is at position 3)
        egraph.parse_and_run_program(None, "(let root (FindUMsb (Const 8)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 3"), "Expected (Const 3), got: {}", result);
    }

    #[test]
    fn test_bitcount() {
        let mut egraph = create_spirv_egraph().unwrap();

        // BitCount(15) = 4 (binary: 1111, four 1-bits)
        egraph.parse_and_run_program(None, "(let root (BitCount (Const 15)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 4"), "Expected (Const 4), got: {}", result);
    }

    #[test]
    fn test_egglog_multiply_primitive_with_zero() {
        // Test if egglog's * primitive works with zero directly (without our rules)
        let mut egraph = EGraph::default();

        // Use the check command to verify multiplication works
        // (check (= result expected))
        let result = egraph.parse_and_run_program(None, "(check (= (* 5 3) 15))");
        assert!(result.is_ok(), "(* 5 3) should equal 15: {:?}", result);

        // Test multiply by zero - this is what's expected to fail
        let result = egraph.parse_and_run_program(None, "(check (= (* 5 0) 0))");
        assert!(result.is_ok(), "(* 5 0) should equal 0: {:?}", result);

        let result = egraph.parse_and_run_program(None, "(check (= (* 0 5) 0))");
        assert!(result.is_ok(), "(* 0 5) should equal 0: {:?}", result);
    }

    #[test]
    fn test_mul_by_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 5 * 0 should fold to 0
        // This tests that the x*0=0 rule fires before strength reduction can cause explosion
        egraph
            .parse_and_run_program(None, r#"(let root (Mul (Const 5) (Const 0)))"#)
            .unwrap();

        // Run just 3 iterations - should be enough for x*0=0 to fire
        egraph
            .parse_and_run_program(None, "(run-schedule (repeat 3 (run)))")
            .unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Mul by zero result: {}", result);
        assert!(
            result.contains("Const 0"),
            "Expected (Const 0), got: {}",
            result
        );
    }

    #[test]
    fn test_mul_by_zero_trace() {
        // Trace what happens in the e-graph step by step
        let mut egraph = create_spirv_egraph().unwrap();

        egraph
            .parse_and_run_program(None, r#"(let root (Mul (Const 5) (Const 0)))"#)
            .unwrap();

        // After initial insertion - what does extract give us?
        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        eprintln!("Initial: {:?}", results);

        // Run 1 iteration
        egraph.parse_and_run_program(None, "(run 1)").unwrap();
        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        eprintln!("After 1 iteration: {:?}", results);

        // Check if we already got 0
        let result = format!("{}", results[0]);
        if result.contains("Const 0") && !result.contains("Mul") {
            eprintln!("SUCCESS: Folded to 0 after 1 iteration");
            return;
        }

        // Run 2nd iteration
        egraph.parse_and_run_program(None, "(run 1)").unwrap();
        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        eprintln!("After 2 iterations: {:?}", results);

        let result = format!("{}", results[0]);
        if result.contains("Const 0") && !result.contains("Mul") {
            eprintln!("SUCCESS: Folded to 0 after 2 iterations");
            return;
        }

        // Run 3rd iteration
        egraph.parse_and_run_program(None, "(run 1)").unwrap();
        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        eprintln!("After 3 iterations: {:?}", results);

        let result = format!("{}", results[0]);
        assert!(
            result.contains("Const 0") && !result.contains("Mul"),
            "Expected (Const 0), got: {}",
            result
        );
    }

    #[test]
    fn test_mul_chain_merge() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (x * 3) * 4 should merge to x * 12
        egraph.parse_and_run_program(None, r#"(let root (Mul (Mul (Sym "x") (Const 3)) (Const 4)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Mul chain result: {}", result);
        assert!(result.contains("12"), "Expected merged constant 12, got: {}", result);
    }

    #[test]
    fn test_add_chain_merge() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (x + 5) + 7 should merge to x + 12
        egraph.parse_and_run_program(None, r#"(let root (Add (Add (Sym "x") (Const 5)) (Const 7)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Add chain result: {}", result);
        assert!(result.contains("12"), "Expected merged constant 12, got: {}", result);
    }

    #[test]
    fn test_bitwise_chain_merge() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (x & 0xFF) & 0x0F should merge to x & 0x0F
        egraph.parse_and_run_program(None, r#"(let root (BitAnd (BitAnd (Sym "x") (Const 255)) (Const 15)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("BitAnd chain result: {}", result);
        // 255 & 15 = 15
        assert!(result.contains("15") && !result.contains("255"),
                "Expected merged constant 15, got: {}", result);
    }

    #[test]
    fn test_gamma_to_min() {
        let mut egraph = create_spirv_egraph().unwrap();

        // select(a < b, a, b) should simplify to min(a, b)
        egraph.parse_and_run_program(None, r#"(let root (Gamma (SLt (Sym "a") (Sym "b")) (Sym "a") (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 15 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Gamma to min result: {}", result);
        assert!(result.contains("SMin"), "Expected SMin, got: {}", result);
    }

    #[test]
    fn test_gamma_to_max() {
        let mut egraph = create_spirv_egraph().unwrap();

        // select(a < b, b, a) should simplify to max(a, b)
        egraph.parse_and_run_program(None, r#"(let root (Gamma (SLt (Sym "a") (Sym "b")) (Sym "b") (Sym "a")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 15 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Gamma to max result: {}", result);
        assert!(result.contains("SMax"), "Expected SMax, got: {}", result);
    }

    #[test]
    fn test_de_morgan() {
        let mut egraph = create_spirv_egraph().unwrap();

        // !(a && b) should equal !a || !b
        egraph.parse_and_run_program(None, r#"(let root (LogNot (LogAnd (Sym "a") (Sym "b"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("De Morgan result: {}", result);
        // Should contain LogOr with LogNot on both operands
        assert!(result.contains("LogOr") || result.contains("LogNot"),
                "Expected De Morgan's law application, got: {}", result);
    }

    #[test]
    fn test_loop_invariant_propagation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Add(LoopInvariant(a), LoopInvariant(b)) should become LoopInvariant(Add(a, b))
        egraph.parse_and_run_program(None, r#"(let root (Add (LoopInvariant (Sym "a")) (LoopInvariant (Sym "b"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Loop invariant result: {}", result);
        // The result should show loop invariant wrapping around the Add
        // After extraction, it might simplify to just (Add a b) due to (LoopInvariant x) = x rule
    }

    #[test]
    fn test_fmul_chain_merge() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (x * 2.0) * 3.0 should merge to x * 6.0
        // f32 bit patterns: 2.0 = 1073741824, 3.0 = 1077936128, 6.0 = 1086324736
        egraph.parse_and_run_program(None, r#"(let root (FMul (FMul (Sym "x") (Const 1073741824)) (Const 1077936128)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("FMul chain result: {}", result);
        // Should contain the bit pattern for 6.0 = 1086324736
        assert!(result.contains("1086324736"), "Expected merged constant for 6.0 (1086324736), got: {}", result);
    }

    #[test]
    fn test_reciprocal_chain() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 1.0 / (1.0 / x) should equal x
        // f32 bit pattern: 1.0 = 1065353216
        egraph.parse_and_run_program(None, r#"(let root (FDiv (Const 1065353216) (FDiv (Const 1065353216) (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Reciprocal chain result: {}", result);
        assert!(result.contains("Sym") && result.contains("x") && !result.contains("FDiv"),
                "Expected just (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_strength_reduction_mul() {
        let mut egraph = create_spirv_egraph().unwrap();

        // x * 8 should be equivalent to x << 3 in the e-graph
        // Both forms are valid; the e-graph knows they're equal
        egraph.parse_and_run_program(None, r#"(let mul_form (Mul (Sym "x") (Const 8)))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let shift_form (Shl (Sym "x") (Const 3)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        // Check that both forms are in the same equivalence class
        let check = egraph.parse_and_run_program(None, "(check (= mul_form shift_form))");
        assert!(check.is_ok(), "Expected mul and shift forms to be equivalent");
    }

    #[test]
    fn test_strength_reduction_div() {
        let mut egraph = create_spirv_egraph().unwrap();

        // x / 4 (unsigned) should be equivalent to x >> 2 in the e-graph
        egraph.parse_and_run_program(None, r#"(let div_form (UDiv (Sym "x") (Const 4)))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let shift_form (ShrU (Sym "x") (Const 2)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        // Check that both forms are in the same equivalence class
        let check = egraph.parse_and_run_program(None, "(check (= div_form shift_form))");
        assert!(check.is_ok(), "Expected div and shift forms to be equivalent");
    }

    #[test]
    fn test_mod_to_and() {
        let mut egraph = create_spirv_egraph().unwrap();

        // x % 8 (unsigned) should be equivalent to x & 7 in the e-graph
        egraph.parse_and_run_program(None, r#"(let mod_form (UMod (Sym "x") (Const 8)))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let and_form (BitAnd (Sym "x") (Const 7)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        // Check that both forms are in the same equivalence class
        let check = egraph.parse_and_run_program(None, "(check (= mod_form and_form))");
        assert!(check.is_ok(), "Expected mod and and forms to be equivalent");
    }

    #[test]
    fn test_double_negation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // --x should become x
        egraph.parse_and_run_program(None, r#"(let root (Neg (Neg (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Double negation result: {}", result);
        assert!(result.contains("Sym") && !result.contains("Neg"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_bitnot_bitnot() {
        let mut egraph = create_spirv_egraph().unwrap();

        // ~~x should become x
        egraph.parse_and_run_program(None, r#"(let root (BitNot (BitNot (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Double BitNot result: {}", result);
        assert!(result.contains("Sym") && !result.contains("BitNot"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_gamma_constant_true() {
        let mut egraph = create_spirv_egraph().unwrap();

        // select(true, a, b) = a
        egraph.parse_and_run_program(None, r#"(let root (Gamma (Const 1) (Sym "a") (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Gamma true result: {}", result);
        assert!(result.contains("Sym") && result.contains("a") && !result.contains("Gamma"),
                "Expected (Sym \"a\"), got: {}", result);
    }

    #[test]
    fn test_gamma_constant_false() {
        let mut egraph = create_spirv_egraph().unwrap();

        // select(false, a, b) = b
        egraph.parse_and_run_program(None, r#"(let root (Gamma (Const 0) (Sym "a") (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Gamma false result: {}", result);
        assert!(result.contains("Sym") && result.contains("b") && !result.contains("Gamma"),
                "Expected (Sym \"b\"), got: {}", result);
    }

    #[test]
    fn test_gamma_same_branches() {
        let mut egraph = create_spirv_egraph().unwrap();

        // select(c, x, x) = x
        egraph.parse_and_run_program(None, r#"(let root (Gamma (Sym "c") (Sym "x") (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Gamma same branches result: {}", result);
        assert!(result.contains("Sym") && result.contains("x") && !result.contains("Gamma"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_clamp_same_bounds() {
        let mut egraph = create_spirv_egraph().unwrap();

        // clamp(x, a, a) = a
        egraph.parse_and_run_program(None, r#"(let root (SClamp (Sym "x") (Sym "a") (Sym "a")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Clamp same bounds result: {}", result);
        assert!(result.contains("Sym") && result.contains("a") && !result.contains("Clamp"),
                "Expected (Sym \"a\"), got: {}", result);
    }

    #[test]
    fn test_sqrt_squared() {
        let mut egraph = create_spirv_egraph().unwrap();

        // sqrt(x) * sqrt(x) = x
        egraph.parse_and_run_program(None, r#"(let root (FMul (Sqrt (Sym "x")) (Sqrt (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Sqrt squared result: {}", result);
        assert!(result.contains("Sym") && result.contains("x") && !result.contains("Sqrt"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_log_exp_cancel() {
        let mut egraph = create_spirv_egraph().unwrap();

        // log(exp(x)) = x
        egraph.parse_and_run_program(None, r#"(let root (Log (Exp (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Log/Exp cancel result: {}", result);
        assert!(result.contains("Sym") && !result.contains("Log") && !result.contains("Exp"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    #[test]
    fn test_fmix_same_args() {
        let mut egraph = create_spirv_egraph().unwrap();

        // mix(a, a, t) = a
        egraph.parse_and_run_program(None, r#"(let root (FMix (Sym "a") (Sym "a") (Sym "t")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("FMix same args result: {}", result);
        assert!(result.contains("Sym") && result.contains("a") && !result.contains("FMix"),
                "Expected (Sym \"a\"), got: {}", result);
    }

    #[test]
    fn test_normalize_normalize() {
        let mut egraph = create_spirv_egraph().unwrap();

        // normalize(normalize(v)) = normalize(v)
        egraph.parse_and_run_program(None, r#"(let root (Normalize (Normalize (Sym "v"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("Double normalize result: {}", result);
        // Should simplify to single Normalize
        let normalize_count = result.matches("Normalize").count();
        assert!(normalize_count == 1, "Expected single Normalize, got: {}", result);
    }

    #[test]
    fn test_x_inversesqrt_x() {
        let mut egraph = create_spirv_egraph().unwrap();

        // x * inversesqrt(x) = sqrt(x)
        egraph.parse_and_run_program(None, r#"(let root (FMul (Sym "x") (InverseSqrt (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("x * inversesqrt(x) result: {}", result);
        assert!(result.contains("Sqrt") && !result.contains("InverseSqrt"),
                "Expected Sqrt, got: {}", result);
    }

    // Tests for generic add/sub cancellation (C++ MergeGenericAddSubArithmetic)
    #[test]
    fn test_add_sub_cancellation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (a - b) + b should simplify to a
        egraph.parse_and_run_program(None, r#"(let expr (Add (Sub (Sym "a") (Sym "b")) (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sym "a"))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected (a - b) + b to simplify to a");
    }

    #[test]
    fn test_sub_add_cancellation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (a + b) - b should simplify to a
        egraph.parse_and_run_program(None, r#"(let expr (Sub (Add (Sym "a") (Sym "b")) (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sym "a"))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected (a + b) - b to simplify to a");
    }

    // Test mask factoring
    #[test]
    fn test_mask_factoring_or() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (a & m) | (b & m) should simplify to (a | b) & m
        egraph.parse_and_run_program(None, r#"(let expr (BitOr (BitAnd (Sym "a") (Sym "m")) (BitAnd (Sym "b") (Sym "m"))))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (BitAnd (BitOr (Sym "a") (Sym "b")) (Sym "m")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected (a & m) | (b & m) to simplify to (a | b) & m");
    }

    // Test FP add/sub cancellation
    #[test]
    fn test_fadd_fsub_cancellation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (a - b) + b should simplify to a for floating point
        egraph.parse_and_run_program(None, r#"(let expr (FAdd (FSub (Sym "a") (Sym "b")) (Sym "b")))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sym "a"))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected FP (a - b) + b to simplify to a");
    }

    // Test FDiv cancellation
    #[test]
    fn test_fdiv_fmul_cancellation() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (y / x) * x should simplify to y
        egraph.parse_and_run_program(None, r#"(let expr (FMul (FDiv (Sym "y") (Sym "x")) (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sym "y"))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected (y / x) * x to simplify to y");
    }

    // Test negate with constant add
    #[test]
    fn test_add_neg_to_sub() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (-x) + c should simplify to c - x
        egraph.parse_and_run_program(None, r#"(let expr (Add (Neg (Sym "x")) (Const 5)))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sub (Const 5) (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 3 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected (-x) + c to simplify to c - x");
    }

    // Test b - (x + a) = (b - a) - x (constant merging)
    #[test]
    fn test_const_sub_add() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 10 - (x + 3) should simplify to 7 - x = (10 - 3) - x
        egraph.parse_and_run_program(None, r#"(let expr (Sub (Const 10) (Add (Sym "x") (Const 3))))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sub (Const 7) (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected 10 - (x + 3) to simplify to 7 - x");
    }

    // Test b - (x - a) = (b + a) - x (constant merging)
    #[test]
    fn test_const_sub_sub() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 10 - (x - 3) should simplify to 13 - x = (10 + 3) - x
        egraph.parse_and_run_program(None, r#"(let expr (Sub (Const 10) (Sub (Sym "x") (Const 3))))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let expected (Sub (Const 13) (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= expr expected))");
        assert!(check.is_ok(), "Expected 10 - (x - 3) to simplify to 13 - x");
    }

    // Test (x + 5) - 5 = x - trace the explosion
    #[test]
    fn test_add_sub_chain_explosion_trace() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (x + 5) - 5 should simplify to x
        egraph.parse_and_run_program(None, r#"(let root (Sub (Add (Sym "x") (Const 5)) (Const 5)))"#).unwrap();

        // Trace iteration by iteration
        for i in 1..=5 {
            let start = std::time::Instant::now();
            egraph.parse_and_run_program(None, "(run 1)").unwrap();
            let elapsed = start.elapsed();

            let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
            let result = format!("{}", results[0]);
            eprintln!("Iter {}: {} ({:?})", i, result, elapsed);

            // If we already got x, we're done
            if result == "(Sym \"x\")" {
                eprintln!("SUCCESS: Simplified to x after {} iterations", i);
                return;
            }

            // If iteration takes too long, we have explosion
            if elapsed.as_millis() > 500 {
                eprintln!("WARNING: Iteration {} took {:?} - possible explosion", i, elapsed);
            }
        }

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Should have simplified to x
        assert!(result == "(Sym \"x\")" || result.contains("Sym") && !result.contains("Add") && !result.contains("Sub"),
                "Expected (Sym \"x\"), got: {}", result);
    }

    // Test real SPIR-V like scenario with multiple expressions
    #[test]
    fn test_add_sub_chain_with_context() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Simulate what the FFI test does: x is a function parameter (id1), 5 is a constant (id2)
        // add = x + 5 (id3), sub = add - 5 (id4)
        // The e-graph will have: id1 = (Sym "id1"), id2 = (Const 5), id3 = (Add id1 id2), id4 = (Sub id3 id2)
        egraph.parse_and_run_program(None, r#"(let id1 (Sym "id1"))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let id2 (Const 5))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let id3 (Add id1 id2))"#).unwrap();
        egraph.parse_and_run_program(None, r#"(let id4 (Sub id3 id2))"#).unwrap();

        // Trace iteration by iteration - run up to 20 like the direct optimizer
        for i in 1..=20 {
            let start = std::time::Instant::now();
            egraph.parse_and_run_program(None, "(run 1)").unwrap();
            let elapsed = start.elapsed();

            let results = egraph.parse_and_run_program(None, "(extract id4)").unwrap();
            let result = format!("{}", results[0]);
            eprintln!("Iter {}: {} ({:?})", i, result, elapsed);

            // If we already got id1 (which should alias to x), we're done
            if result.contains("Sym") && result.contains("id1") && !result.contains("Add") && !result.contains("Sub") {
                eprintln!("SUCCESS: Simplified to id1 after {} iterations", i);
                return;
            }

            // If iteration takes too long, we have explosion
            if elapsed.as_secs() > 5 {
                panic!("EXPLOSION: Iteration {} took {:?}", i, elapsed);
            }
        }

        let results = egraph.parse_and_run_program(None, "(extract id4)").unwrap();
        let result = format!("{}", results[0]);
        // Should have simplified to id1
        assert!(result.contains("Sym") && result.contains("id1") && !result.contains("Add") && !result.contains("Sub"),
                "Expected (Sym \"id1\"), got: {}", result);
    }

    // Test 4*2 + 4*3 = 20 - trace the explosion
    #[test]
    fn test_linear_combination_explosion_trace() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 4*2 + 4*3 should fold to 20 (via 8 + 12)
        egraph.parse_and_run_program(None, r#"(let root (Add (Mul (Const 4) (Const 2)) (Mul (Const 4) (Const 3))))"#).unwrap();

        // Trace iteration by iteration
        for i in 1..=10 {
            let start = std::time::Instant::now();
            egraph.parse_and_run_program(None, "(run 1)").unwrap();
            let elapsed = start.elapsed();

            let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
            let result = format!("{}", results[0]);
            eprintln!("Iter {}: {} ({:?})", i, result, elapsed);

            // Check if we got a constant
            if result == "(Const 20)" || result.contains("20") {
                eprintln!("SUCCESS: Folded to 20 after {} iterations", i);
                return;
            }

            // If iteration takes too long, we have explosion
            if elapsed.as_millis() > 500 {
                eprintln!("WARNING: Iteration {} took {:?} - possible explosion", i, elapsed);
                // Don't continue if it's exploding
                break;
            }
        }

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Should have folded to 20
        assert!(result.contains("20"), "Expected constant 20, got: {}", result);
    }

    // =============================================================================
    // GLSL Constant Folding Tests
    // =============================================================================

    #[test]
    fn test_sin_constant_fold() {
        let mut egraph = create_spirv_egraph().unwrap();

        // sin(0) should fold to 0
        egraph.parse_and_run_program(None, "(let root (Sin (Const 0)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Result should be a constant (sin(0) = 0)
        assert!(result.contains("Const"), "sin(0) should fold to a constant, got: {}", result);
    }

    #[test]
    fn test_exp_log_constant_fold() {
        let mut egraph = create_spirv_egraph().unwrap();

        // exp(0) should fold to 1 (as float bits)
        // 1.0f64.to_bits() = 4607182418800017408
        egraph.parse_and_run_program(None, "(let root (Exp (Const 0)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Result should be a constant
        assert!(result.contains("Const"), "exp(0) should fold to a constant, got: {}", result);
    }

    #[test]
    fn test_sqrt_constant_fold() {
        let mut egraph = create_spirv_egraph().unwrap();

        // sqrt(4.0) should fold to 2.0
        // 4.0f64.to_bits() = 4616189618054758400
        let four_bits = 4.0_f64.to_bits() as i64;
        let expr = format!("(let root (Sqrt (Const {})))", four_bits);
        egraph.parse_and_run_program(None, &expr).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Result should be a constant
        assert!(result.contains("Const"), "sqrt(4.0) should fold to a constant, got: {}", result);
    }

    // =============================================================================
    // Clamp Feeding Compare Tests
    // =============================================================================

    #[test]
    fn test_clamp_lt_lo_is_false() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FClamp(x, lo, hi) < lo => false (Const 0)
        egraph.parse_and_run_program(None, r#"(let root (FOrdLt (FClamp (Sym "x") (Sym "lo") (Sym "hi")) (Sym "lo")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"), "clamp(x, lo, hi) < lo should be false, got: {}", result);
    }

    #[test]
    fn test_clamp_ge_lo_is_true() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FClamp(x, lo, hi) >= lo => true (Const 1)
        egraph.parse_and_run_program(None, r#"(let root (FOrdGe (FClamp (Sym "x") (Sym "lo") (Sym "hi")) (Sym "lo")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 1"), "clamp(x, lo, hi) >= lo should be true, got: {}", result);
    }

    #[test]
    fn test_clamp_gt_hi_is_false() {
        let mut egraph = create_spirv_egraph().unwrap();

        // FClamp(x, lo, hi) > hi => false (Const 0)
        egraph.parse_and_run_program(None, r#"(let root (FOrdGt (FClamp (Sym "x") (Sym "lo") (Sym "hi")) (Sym "hi")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"), "clamp(x, lo, hi) > hi should be false, got: {}", result);
    }

    // =============================================================================
    // Div-Mul Cancellation Tests
    // =============================================================================

    #[test]
    fn test_div_mul_cancel() {
        let mut egraph = create_spirv_egraph().unwrap();

        // (y / x) * x should simplify to y
        egraph.parse_and_run_program(None, r#"(let root (Mul (SDiv (Sym "y") (Sym "x")) (Sym "x")))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Should simplify to just y
        assert!(result.contains("Sym") && result.contains("y") && !result.contains("SDiv"),
                "(y / x) * x should simplify to y, got: {}", result);
    }

    #[test]
    fn test_mul_div_cancel_unsigned() {
        let mut egraph = create_spirv_egraph().unwrap();

        // x * (y / x) should simplify to y
        egraph.parse_and_run_program(None, r#"(let root (Mul (Sym "x") (UDiv (Sym "y") (Sym "x"))))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Should simplify to just y
        assert!(result.contains("Sym") && result.contains("y") && !result.contains("UDiv"),
                "x * (y / x) should simplify to y, got: {}", result);
    }

    /// Test that identity operations don't cause e-graph explosion
    #[test]
    fn test_identity_no_explosion() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Add with Const 0 - this used to cause explosion due to associativity rule interaction
        egraph.parse_and_run_program(None, "(let c5 (Const 5))").unwrap();
        egraph.parse_and_run_program(None, "(let c0 (Const 0))").unwrap();
        egraph.parse_and_run_program(None, "(let expr (Add c5 c0))").unwrap(); // 5 + 0 = 5 (identity)

        // Run 20 iterations - should complete quickly
        egraph.parse_and_run_program(None, "(run-schedule (repeat 20 (run)))").unwrap();

        // Extract - should be (Const 5)
        let results = egraph.parse_and_run_program(None, "(extract c5)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("5"), "Expected Const 5, got: {}", result);
    }

    // =========================================================================
    // Memory Operation Tests
    // =========================================================================

    #[test]
    fn test_load_after_store_forwarding() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Load from a pointer that was just stored to should return the stored value
        // Load(ptr, StoreMem(ptr, val, prev)) = val
        egraph.parse_and_run_program(None, r#"
            (let ptr (Var "x" 0))
            (let val (Const 42))
            (let mem (StoreMem ptr val (InitMem)))
            (let root (Load ptr mem))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 42"),
                "Load after store should forward value, got: {}", result);
    }

    #[test]
    fn test_dead_store_elimination() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Store-after-store: second store kills the first
        // StoreMem(ptr, val2, StoreMem(ptr, val1, prev)) = StoreMem(ptr, val2, prev)
        egraph.parse_and_run_program(None, r#"
            (let ptr (Var "x" 0))
            (let inner (StoreMem ptr (Const 10) (InitMem)))
            (let root (StoreMem ptr (Const 20) inner))
            (let expected (StoreMem ptr (Const 20) (InitMem)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        // Check that root equals expected (dead store eliminated)
        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "Dead store elimination should work");
    }

    #[test]
    fn test_merge_mem_same_branches() {
        let mut egraph = create_spirv_egraph().unwrap();

        // MergeMem with same state on both branches = that state
        egraph.parse_and_run_program(None, r#"
            (let mem (InitMem))
            (let root (MergeMem (Sym "cond") mem mem))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("InitMem") && !result.contains("MergeMem"),
                "MergeMem with same branches should simplify, got: {}", result);
    }

    #[test]
    fn test_merge_mem_constant_condition() {
        let mut egraph = create_spirv_egraph().unwrap();

        // MergeMem with constant true condition = then branch
        egraph.parse_and_run_program(None, r#"
            (let then_mem (StoreMem (Var "x" 0) (Const 1) (InitMem)))
            (let else_mem (StoreMem (Var "y" 0) (Const 2) (InitMem)))
            (let root (MergeMem (Const 1) then_mem else_mem))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root then_mem))");
        assert!(check.is_ok(), "MergeMem with true condition should select then branch");
    }

    #[test]
    fn test_access_chain_load_store_forwarding() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Load through access chain from store through same access chain
        egraph.parse_and_run_program(None, r#"
            (let base (Var "arr" 0))
            (let chain (AccessChain1 base 5))
            (let mem (StoreMem chain (Const 99) (InitMem)))
            (let root (Load chain mem))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 99"),
                "Access chain load-store forwarding should work, got: {}", result);
    }

    // =========================================================================
    // Function Inlining Tests
    // =========================================================================

    #[test]
    fn test_subst_arg_replacement() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Subst(Arg(0), 0, val) = val
        egraph.parse_and_run_program(None, r#"
            (let val (Const 42))
            (let root (Subst (Arg 0) 0 val))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 42"),
                "Subst(Arg(0), 0, val) should equal val, got: {}", result);
    }

    #[test]
    fn test_subst_constant_unchanged() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Subst(Const c, idx, val) = Const c (constants don't change)
        egraph.parse_and_run_program(None, r#"
            (let root (Subst (Const 100) 0 (Const 999)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 100") && !result.contains("999"),
                "Subst on constant should be unchanged, got: {}", result);
    }

    #[test]
    fn test_subst_into_binary_op() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Subst(Add(Arg(0), Const(5)), 0, Const(10)) = Add(Const(10), Const(5)) = Const(15)
        egraph.parse_and_run_program(None, r#"
            (let body (Add (Arg 0) (Const 5)))
            (let root (Subst body 0 (Const 10)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        // Should fold to Const 15
        assert!(result.contains("Const 15"),
                "Subst should propagate and fold, got: {}", result);
    }

    // =========================================================================
    // Loop Optimization Tests
    // =========================================================================

    #[test]
    fn test_bounded_theta_zero_iterations() {
        let mut egraph = create_spirv_egraph().unwrap();

        // BoundedTheta with 0 iterations returns init
        egraph.parse_and_run_program(None, r#"
            (let init (Const 42))
            (let body (Add (Arg 0) (Const 1)))
            (let root (BoundedTheta 0 body init))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 42"),
                "BoundedTheta(0, ...) should return init, got: {}", result);
    }

    #[test]
    fn test_theta_constant_false_condition() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Theta with constant false condition returns init
        egraph.parse_and_run_program(None, r#"
            (let init (Const 100))
            (let root (Theta (Const 0) (Add (LoopVar) (Const 1)) init))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 100"),
                "Theta with false condition should return init, got: {}", result);
    }

    #[test]
    fn test_loop_strength_reduction_mul_to_shift() {
        let mut egraph = create_spirv_egraph().unwrap();

        // LoopVar * 4 should become LoopVar << 2
        egraph.parse_and_run_program(None, r#"
            (let root (Mul (LoopVar) (Const 4)))
            (let expected (Shl (LoopVar) (Const 2)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "LoopVar * 4 should equal LoopVar << 2");
    }

    #[test]
    fn test_loop_invariant_propagation_extended() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Operations on loop-invariant values should be loop-invariant
        egraph.parse_and_run_program(None, r#"
            (let a (LoopInvariant (Const 10)))
            (let b (LoopInvariant (Const 20)))
            (let root (Add a b))
            (let expected (LoopInvariant (Add (Const 10) (Const 20))))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "Add of loop-invariants should be loop-invariant");
    }

    #[test]
    fn test_loop_iter_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // LoopIter(0, x) = x
        egraph.parse_and_run_program(None, r#"
            (let val (Const 77))
            (let root (LoopIter 0 val))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 77"),
                "LoopIter(0, x) should equal x, got: {}", result);
    }

    // =========================================================================
    // Specialization Constant Tests
    // =========================================================================

    #[test]
    fn test_spec_add_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // SpecAdd(x, 0) = x
        egraph.parse_and_run_program(None, r#"
            (let spec (SpecConst 0 42))
            (let root (SpecAdd spec (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("SpecConst") && !result.contains("SpecAdd"),
                "SpecAdd(x, 0) should simplify to x, got: {}", result);
    }

    #[test]
    fn test_spec_mul_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // SpecMul(x, 0) = 0
        egraph.parse_and_run_program(None, r#"
            (let spec (SpecConst 0 42))
            (let root (SpecMul spec (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"),
                "SpecMul(x, 0) should be 0, got: {}", result);
    }

    #[test]
    fn test_spec_select_constant_condition() {
        let mut egraph = create_spirv_egraph().unwrap();

        // SpecSelect(true, a, b) = a
        egraph.parse_and_run_program(None, r#"
            (let a (Const 10))
            (let b (Const 20))
            (let root (SpecSelect (Const 1) a b))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 10"),
                "SpecSelect(true, a, b) should be a, got: {}", result);
    }

    #[test]
    fn test_spec_eq_self() {
        let mut egraph = create_spirv_egraph().unwrap();

        // SpecEq(x, x) = 1 (true)
        egraph.parse_and_run_program(None, r#"
            (let spec (SpecConst 0 42))
            (let root (SpecEq spec spec))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 1"),
                "SpecEq(x, x) should be true (1), got: {}", result);
    }

    #[test]
    fn test_spec_strength_reduction() {
        let mut egraph = create_spirv_egraph().unwrap();

        // SpecMul(x, 4) should become SpecShl(x, 2)
        egraph.parse_and_run_program(None, r#"
            (let spec (SpecConst 0 42))
            (let root (SpecMul spec (Const 4)))
            (let expected (SpecShl spec (Const 2)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "SpecMul(x, 4) should equal SpecShl(x, 2)");
    }

    // =========================================================================
    // Derivative Operation Tests
    // =========================================================================

    #[test]
    fn test_derivative_of_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // DPdx(Const c) = 0
        egraph.parse_and_run_program(None, r#"
            (let root (DPdx (Const 42)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"),
                "DPdx of constant should be 0, got: {}", result);
    }

    #[test]
    fn test_fwidth_of_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Fwidth(Const c) = 0
        egraph.parse_and_run_program(None, r#"
            (let root (Fwidth (Const 42)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"),
                "Fwidth of constant should be 0, got: {}", result);
    }

    // =========================================================================
    // Subgroup/Wave Operation Tests
    // =========================================================================

    #[test]
    fn test_group_broadcast_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupBroadcast(Const c, lane) = Const c
        egraph.parse_and_run_program(None, r#"
            (let root (GroupBroadcast (Const 42) (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 42") && !result.contains("GroupBroadcast"),
                "GroupBroadcast of constant should be that constant, got: {}", result);
    }

    #[test]
    fn test_group_broadcast_first_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupBroadcastFirst(Const c) = Const c
        egraph.parse_and_run_program(None, r#"
            (let root (GroupBroadcastFirst (Const 123)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 123") && !result.contains("GroupBroadcastFirst"),
                "GroupBroadcastFirst of constant should be that constant, got: {}", result);
    }

    #[test]
    fn test_group_all_constant_true() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupAll(true) = true
        egraph.parse_and_run_program(None, r#"
            (let root (GroupAll (Const 1)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 1") && !result.contains("GroupAll"),
                "GroupAll(true) should be true, got: {}", result);
    }

    #[test]
    fn test_group_any_constant_false() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupAny(false) = false
        egraph.parse_and_run_program(None, r#"
            (let root (GroupAny (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0") && !result.contains("GroupAny"),
                "GroupAny(false) should be false, got: {}", result);
    }

    #[test]
    fn test_group_all_equal_constant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupAllEqual(Const c) = true (all lanes have same constant)
        egraph.parse_and_run_program(None, r#"
            (let root (GroupAllEqual (Const 42)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 1"),
                "GroupAllEqual of constant should be true, got: {}", result);
    }

    #[test]
    fn test_group_shuffle_xor_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupShuffleXor(x, 0) = x (identity shuffle)
        egraph.parse_and_run_program(None, r#"
            (let val (Sym "x"))
            (let root (GroupShuffleXor val (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root val))");
        assert!(check.is_ok(), "GroupShuffleXor(x, 0) should equal x");
    }

    #[test]
    fn test_group_iadd_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupIAdd(0) = 0 (sum of zeros)
        egraph.parse_and_run_program(None, r#"
            (let root (GroupIAdd (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"),
                "GroupIAdd(0) should be 0, got: {}", result);
    }

    #[test]
    fn test_group_imul_one() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupIMul(1) = 1 (product of ones)
        egraph.parse_and_run_program(None, r#"
            (let root (GroupIMul (Const 1)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 1"),
                "GroupIMul(1) should be 1, got: {}", result);
    }

    #[test]
    fn test_group_bit_or_zero() {
        let mut egraph = create_spirv_egraph().unwrap();

        // GroupBitOr(0) = 0 (OR of zeros)
        egraph.parse_and_run_program(None, r#"
            (let root (GroupBitOr (Const 0)))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        let result = format!("{}", results[0]);
        assert!(result.contains("Const 0"),
                "GroupBitOr(0) should be 0, got: {}", result);
    }

    // =========================================================================
    // Access Chain Optimization Tests
    // =========================================================================

    #[test]
    fn test_access_chain_combining() {
        let mut egraph = create_spirv_egraph().unwrap();

        // AccessChain1(AccessChain1(base, i1), i2) = AccessChain2(base, i1, i2)
        egraph.parse_and_run_program(None, r#"
            (let base (Var "arr" 0))
            (let root (AccessChain1 (AccessChain1 base 0) 1))
            (let expected (AccessChain2 base 0 1))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "Nested AccessChain1 should combine into AccessChain2");
    }

    #[test]
    fn test_access_chain_combining_three_levels() {
        let mut egraph = create_spirv_egraph().unwrap();

        // AccessChain1(AccessChain2(base, i1, i2), i3) = AccessChain3(base, i1, i2, i3)
        egraph.parse_and_run_program(None, r#"
            (let base (Var "struct" 0))
            (let root (AccessChain1 (AccessChain2 base 0 1) 2))
            (let expected (AccessChain3 base 0 1 2))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "AccessChain1 of AccessChain2 should combine into AccessChain3");
    }

    #[test]
    fn test_dynamic_access_chain_const() {
        let mut egraph = create_spirv_egraph().unwrap();

        // AccessChainDyn(base, Const i) = AccessChain1(base, i)
        egraph.parse_and_run_program(None, r#"
            (let base (Var "arr" 0))
            (let root (AccessChainDyn base (Const 5)))
            (let expected (AccessChain1 base 5))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "AccessChainDyn with constant should become AccessChain1");
    }

    #[test]
    fn test_load_loop_invariant() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Load from loop-invariant pointer and memory is loop-invariant
        egraph.parse_and_run_program(None, r#"
            (let ptr (LoopInvariant (Var "p" 0)))
            (let mem (LoopInvariant (InitMem)))
            (let root (Load ptr mem))
            (let inner_load (Load (Var "p" 0) (InitMem)))
            (let expected (LoopInvariant inner_load))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 5 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "Load from loop-invariant ptr and mem should be loop-invariant");
    }

    #[test]
    fn test_dead_store_across_branches() {
        let mut egraph = create_spirv_egraph().unwrap();

        // Store followed by MergeMem where both branches overwrite eliminates first store
        egraph.parse_and_run_program(None, r#"
            (let ptr (Var "p" 0))
            (let cond (Sym "c"))
            (let prev (InitMem))
            (let inner (StoreMem ptr (Const 0) prev))
            (let branch1 (StoreMem ptr (Const 1) inner))
            (let branch2 (StoreMem ptr (Const 2) inner))
            (let root (MergeMem cond branch1 branch2))
            (let expected_b1 (StoreMem ptr (Const 1) prev))
            (let expected_b2 (StoreMem ptr (Const 2) prev))
            (let expected (MergeMem cond expected_b1 expected_b2))
        "#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let check = egraph.parse_and_run_program(None, "(check (= root expected))");
        assert!(check.is_ok(), "Dead store before branch should be eliminated");
    }

}
