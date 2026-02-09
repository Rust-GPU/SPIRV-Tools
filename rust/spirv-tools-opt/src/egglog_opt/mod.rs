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
//! - `mem2reg.egg`: Memory to register promotion
//! - `legalization.egg`: Target-specific legalization (Vulkan/OpenCL)
//! - `licm.egg`: Loop invariant code motion
//! - `loop_fusion.egg`: Loop fusion and fission
//! - `instrumentation.egg`: Profiling and debugging hooks
//! - `merge_return.egg`: Merge-return and control flow flattening

mod primitives;

use egglog::sort::{F, OrderedFloat};
use egglog::{add_primitive, EGraph};
// The primitive functions are used inside `add_primitive!` macro closures, which
// the compiler doesn't track as direct usage of the imported names.
#[allow(unused_imports)]
use primitives::*;

/// The egglog program defining the SPIR-V RVSDG language and rewrite rules.
/// This is assembled from multiple rule files at compile time.
const SPIRV_EGGLOG_PROGRAM: &str = concat!(
    include_str!("../rules/datatypes.egg"),
    "\n",
    include_str!("../rules/rvsdg.egg"),
    "\n",
    include_str!("../rules/arithmetic.egg"),
    "\n",
    include_str!("../rules/bitwise.egg"),
    "\n",
    include_str!("../rules/comparison.egg"),
    "\n",
    include_str!("../rules/logical.egg"),
    "\n",
    include_str!("../rules/floating_point.egg"),
    "\n",
    include_str!("../rules/vector.egg"),
    "\n",
    include_str!("../rules/matrix.egg"),
    "\n",
    include_str!("../rules/glsl.egg"),
    "\n",
    include_str!("../rules/type_conversion.egg"),
    "\n",
    include_str!("../rules/constant_folding.egg"),
    "\n",
    include_str!("../rules/memory.egg"),
    "\n",
    include_str!("../rules/inlining.egg"),
    "\n",
    include_str!("../rules/loop_unroll.egg"),
    "\n",
    include_str!("../rules/spec_constant.egg"),
    "\n",
    include_str!("../rules/sroa.egg"),
    "\n",
    include_str!("../rules/advanced_loops.egg"),
    "\n",
    include_str!("../rules/copy_propagation.egg"),
    "\n",
    include_str!("../rules/graphics.egg"),
    "\n",
    include_str!("../rules/float_conversion.egg"),
    "\n",
    include_str!("../rules/cleanup.egg"),
    "\n",
    include_str!("../rules/subgroup.egg"),
    "\n",
    include_str!("../rules/bitfield.egg"),
    "\n",
    include_str!("../rules/dce.egg"),
    "\n",
    include_str!("../rules/licm.egg"),
    "\n",
    include_str!("../rules/mem2reg.egg"),
    // NOTE: merge_return.egg rules are ALREADY in rvsdg.egg (lines 213, 230).
    //       Enabling it causes egglog "rule was already present" errors.
    // NOTE: The following files have structural issues that prevent enabling:
    // - loop_fusion.egg: Uses undefined types (Pair, Triple, CountRange, etc.) and
    //   has type mismatches (Seq takes Effect Effect, but Theta returns Expr)
    // - legalization.egg: Uses many undefined types (SDiv8, FAdd16, MakePair64, etc.)
    // - instrumentation.egg: Fundamental type mismatches (Seq used with Expr instead
    //   of Effect, AtomicIAdd called with 2 args instead of 3)
    // "\n",
    // include_str!("../rules/merge_return.egg"),
    // "\n",
    // include_str!("../rules/loop_fusion.egg"),
    // "\n",
    // include_str!("../rules/legalization.egg"),
    // "\n",
    // include_str!("../rules/instrumentation.egg"),
);

/// Rules that use custom primitives (must be loaded after primitives are registered).
const SPIRV_EGGLOG_PRIMITIVES_PROGRAM: &str = include_str!("../rules/primitives.egg");

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
    if (a & b) == 0 {
        Some(())
    } else {
        None
    }
}

/// Check if left shift would clear all bits in a mask.
fn shl_clears_mask(mask: i64, shift: i64) -> Option<()> {
    let shift = shift as u32;
    if shift >= 64 {
        return Some(());
    }
    let surviving_mask = !((1i64 << shift) - 1);
    if (mask & surviving_mask) == 0 {
        Some(())
    } else {
        None
    }
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
    if (mask & surviving_mask) == 0 {
        Some(())
    } else {
        None
    }
}

/// Check if mask a is a superset of mask b.
fn mask_superset(a: i64, b: i64) -> Option<()> {
    if (a & b) == b {
        Some(())
    } else {
        None
    }
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

/// Check if an integer, when interpreted as a f64, has an exact reciprocal.
/// A float has an exact reciprocal iff it is a power of 2 (mantissa bits all zero).
fn has_exact_recip(x: i64) -> Option<()> {
    let bits = x as u64;
    let f = f64::from_bits(bits);
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    // f64 mantissa is 52 bits. A power of 2 has all mantissa bits zero.
    const F64_MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    if (bits & F64_MANTISSA_MASK) == 0 {
        let recip = 1.0 / f;
        if recip.is_finite() { Some(()) } else { None }
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
    if f == 1.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 0.0 bit pattern.
fn is_float_zero32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 -1.0 bit pattern.
fn is_float_neg_one32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -1.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 2.0 bit pattern.
fn is_float_two32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 2.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 3.0 bit pattern (for powi(3) optimization).
fn is_float_three32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 3.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 4.0 bit pattern (for powi(4) optimization).
fn is_float_four32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 4.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 0.5 bit pattern (for sqrt optimizations).
fn is_float_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.5 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 -0.5 bit pattern (for inverse sqrt optimizations).
fn is_float_neg_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -0.5 {
        Some(())
    } else {
        None
    }
}

/// Check if an f32 constant (stored as i64 bit pattern) has an exact reciprocal.
/// A float has an exact reciprocal iff it is a power of 2 (mantissa bits all zero).
fn has_exact_recip32(x: i64) -> Option<()> {
    let bits = x as u32;
    let f = f32::from_bits(bits);
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    // f32 mantissa is 23 bits. A power of 2 has all mantissa bits zero.
    const F32_MANTISSA_MASK: u32 = (1u32 << 23) - 1;
    if (bits & F32_MANTISSA_MASK) == 0 {
        let recip = 1.0f32 / f;
        if recip.is_finite() { Some(()) } else { None }
    } else {
        None
    }
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
// FConst uses native f64 values, so primitives operate on f64 directly.

/// Auto-detect f32/f64 bit pattern and multiply.
/// If both values have zero high 32 bits, treat as f32; otherwise f64.
/// Returns the product as a native f64.
fn float_mul_auto(a: i64, b: i64) -> f64 {
    if (a >> 32) == 0 && (b >> 32) == 0 {
        let fa = f32::from_bits(a as u32);
        let fb = f32::from_bits(b as u32);
        (fa * fb) as f64
    } else {
        let fa = f64::from_bits(a as u64);
        let fb = f64::from_bits(b as u64);
        fa * fb
    }
}

/// Auto-detect f32/f64 bit pattern binary operation, returning result as i64 bit pattern.
/// If both values have zero high 32 bits, treat as f32; otherwise f64.
fn float_binop_bits(a: i64, b: i64, op32: fn(f32, f32) -> f32, op64: fn(f64, f64) -> f64) -> i64 {
    if (a >> 32) == 0 && (b >> 32) == 0 {
        let r = op32(f32::from_bits(a as u32), f32::from_bits(b as u32));
        r.to_bits() as i64
    } else {
        let r = op64(f64::from_bits(a as u64), f64::from_bits(b as u64));
        r.to_bits() as i64
    }
}

/// Auto-detect f32/f64 bit pattern and negate, returning result as i64 bit pattern.
fn float_neg_bits(a: i64) -> i64 {
    if (a >> 32) == 0 {
        let r = -f32::from_bits(a as u32);
        r.to_bits() as i64
    } else {
        let r = -f64::from_bits(a as u64);
        r.to_bits() as i64
    }
}

/// Bitcast 32-bit integer constant to f64 (reinterpret lower 32 bits as f32, promote to f64).
fn bitcast_int_to_float32(v: i64) -> f64 {
    f32::from_bits(v as u32) as f64
}

/// Bitcast 64-bit integer constant to f64 (reinterpret bits as f64).
fn bitcast_int_to_float64(v: i64) -> f64 {
    f64::from_bits(v as u64)
}

/// Bitcast f64 constant to 32-bit integer (demote to f32, get u32 bits).
fn bitcast_float_to_int32(v: f64) -> i64 {
    (v as f32).to_bits() as i64
}

/// Bitcast f64 constant to 64-bit integer (get u64 bits).
fn bitcast_float_to_int64(v: f64) -> i64 {
    v.to_bits() as i64
}

/// SConvert constant fold: sign-extend or truncate to target width.
fn sconvert_fold(v: i64, dst_width: i64) -> i64 {
    let dw = dst_width as u32;
    if dw >= 64 {
        v
    } else {
        let shift = 64 - dw;
        (v << shift) >> shift
    }
}

/// UConvert constant fold: zero-extend (mask source) then truncate to target.
fn uconvert_fold(v: i64, src_width: i64, dst_width: i64) -> i64 {
    let sw = src_width as u32;
    let unsigned = if sw >= 64 { v } else { v & ((1i64 << sw) - 1) };
    let dw = dst_width as u32;
    if dw >= 64 {
        unsigned
    } else {
        unsigned & ((1i64 << dw) - 1)
    }
}

/// Quantize a float value to IEEE 754 half-precision (f16), returning f64.
/// Preserves NaN, Inf. Overflows to Inf, underflows to zero.
/// SPIR-V spec: result is value of operand rounded to f16 precision.
fn quantize_to_f16(v: f64) -> f64 {
    if v.is_nan() {
        return f64::NAN;
    }
    if v.is_infinite() {
        return v;
    }
    // Convert to f32 first (lossless for f16 range), then quantize to f16
    let f = v as f32;
    let bits = f.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;

    // f16: 1 sign, 5 exponent (bias 15), 10 mantissa
    // f32: 1 sign, 8 exponent (bias 127), 23 mantissa
    let f16_bits: u16 = if exp == 0 {
        // f32 denormal → f16 zero (too small for f16)
        (sign as u16) << 15
    } else if exp == 0xFF {
        // f32 inf/nan (already handled above, but for safety)
        if frac == 0 {
            ((sign as u16) << 15) | 0x7C00
        } else {
            0x7E00 // quiet NaN
        }
    } else {
        let new_exp = exp - 127 + 15; // rebias from f32 to f16
        if new_exp >= 31 {
            // overflow → infinity
            ((sign as u16) << 15) | 0x7C00
        } else if new_exp <= 0 {
            // underflow → denormal or zero
            if new_exp < -10 {
                // too small even for f16 denormal
                (sign as u16) << 15
            } else {
                // f16 denormal: implicit 1 + fraction, shifted right
                let full_frac = frac | 0x0080_0000; // add implicit 1
                let shift = (1 - new_exp) as u32 + 13; // 13 = 23 - 10 mantissa diff
                let half = 1u32 << (shift - 1);
                let rounded = (full_frac + half) >> shift;
                ((sign as u16) << 15) | (rounded as u16)
            }
        } else {
            // normal f16: round mantissa from 23 to 10 bits
            let half = 1u32 << 12; // round bit
            let rounded_frac = (frac + half) >> 13;
            if rounded_frac >= 0x400 {
                // mantissa overflow, increment exponent
                ((sign as u16) << 15) | (((new_exp + 1) as u16) << 10)
            } else {
                ((sign as u16) << 15) | ((new_exp as u16) << 10) | (rounded_frac as u16)
            }
        }
    };

    // Convert f16 back to f64
    let f16_sign = ((f16_bits >> 15) & 1) as u64;
    let f16_exp = ((f16_bits >> 10) & 0x1F) as i32;
    let f16_frac = (f16_bits & 0x03FF) as u64;

    let f64_bits = if f16_exp == 0 {
        if f16_frac == 0 {
            // zero
            f16_sign << 63
        } else {
            // denormal: convert to normalized f64
            let mut e = -14i32;
            let mut m = f16_frac;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove implicit 1
            let f64_exp = ((e + 1023) as u64) & 0x7FF;
            (f16_sign << 63) | (f64_exp << 52) | (m << 42)
        }
    } else if f16_exp == 31 {
        if f16_frac == 0 {
            // infinity
            (f16_sign << 63) | (0x7FFu64 << 52)
        } else {
            // NaN
            (f16_sign << 63) | (0x7FFu64 << 52) | (1u64 << 51)
        }
    } else {
        // normal
        let f64_exp = ((f16_exp - 15 + 1023) as u64) & 0x7FF;
        (f16_sign << 63) | (f64_exp << 52) | (f16_frac << 42)
    };

    f64::from_bits(f64_bits)
}

/// NaN-aware minimum: if a is NaN return b, if b is NaN return a, else min.
fn float_nmin(a: f64, b: f64) -> f64 {
    if a.is_nan() { b } else if b.is_nan() { a } else { a.min(b) }
}

/// NaN-aware maximum: if a is NaN return b, if b is NaN return a, else max.
fn float_nmax(a: f64, b: f64) -> f64 {
    if a.is_nan() { b } else if b.is_nan() { a } else { a.max(b) }
}

/// NaN-aware clamp: nmax(nmin(x, hi), lo).
fn float_nclamp(x: f64, lo: f64, hi: f64) -> f64 {
    float_nmax(float_nmin(x, hi), lo)
}

/// Compute sign of an integer constant (-1, 0, or 1).
fn int_sign(x: i64) -> i64 {
    if x > 0 { 1 } else if x < 0 { -1 } else { 0 }
}

/// Compute sign of a float constant (-1.0, 0.0, or 1.0).
fn float_sign_f64(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

/// Compute fract (fractional part) of a float constant.
fn float_fract_f64(x: f64) -> f64 {
    x - x.floor()
}

/// Compute mix (linear interpolation) of float constants.
fn float_mix_f64(x: f64, y: f64, a: f64) -> f64 {
    x * (1.0 - a) + y * a
}

/// Compute smoothstep function.
fn float_smoothstep_f64(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
    add_primitive!(
        &mut egraph,
        "bitrev32" = |a: i64| -> i64 { bitreverse32(a) }
    );
    add_primitive!(
        &mut egraph,
        "bitrev64" = |a: i64| -> i64 { bitreverse64(a) }
    );

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
    add_primitive!(&mut egraph, "find-lsb" = |a: i64| -> i64 { find_lsb(a) });
    add_primitive!(
        &mut egraph,
        "find-msb-unsigned" = |a: i64| -> i64 { find_msb_unsigned(a) }
    );
    add_primitive!(
        &mut egraph,
        "find-msb-signed" = |a: i64| -> i64 { find_msb_signed(a) }
    );
    add_primitive!(&mut egraph, "popcount" = |a: i64| -> i64 { popcount(a) });

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
    add_primitive!(&mut egraph, "log2-pow2" = |a: i64| -> i64 { log2_pow2(a) });

    // Float reciprocal primitives for x/c -> x*(1/c) optimization (f64 versions, legacy)
    add_primitive!(&mut egraph, "has-exact-recip" = |a: i64| -?> () {
        has_exact_recip(a)
    });
    add_primitive!(
        &mut egraph,
        "float-recip" = |a: i64| -> i64 { float_recip(a) }
    );

    // f32 arithmetic primitives for FP constant folding
    // These interpret i64 as f32 bit patterns (lower 32 bits)
    add_primitive!(
        &mut egraph,
        "float-mul32" = |a: i64, b: i64| -> i64 { float_mul32(a, b) }
    );
    add_primitive!(&mut egraph, "float-div32" = |a: i64, b: i64| -?> i64 { float_div32(a, b) });
    add_primitive!(
        &mut egraph,
        "float-add32" = |a: i64, b: i64| -> i64 { float_add32(a, b) }
    );
    add_primitive!(
        &mut egraph,
        "float-sub32" = |a: i64, b: i64| -> i64 { float_sub32(a, b) }
    );
    add_primitive!(
        &mut egraph,
        "float-neg32" = |a: i64| -> i64 { float_neg32(a) }
    );

    // f32 identity check primitives
    add_primitive!(&mut egraph, "is-float-one32" = |a: i64| -?> () { is_float_one32(a) });
    add_primitive!(&mut egraph, "is-float-zero32" = |a: i64| -?> () { is_float_zero32(a) });
    add_primitive!(&mut egraph, "is-float-neg-one32" = |a: i64| -?> () { is_float_neg_one32(a) });
    add_primitive!(&mut egraph, "is-float-two32" = |a: i64| -?> () { is_float_two32(a) });
    add_primitive!(&mut egraph, "is-float-three32" = |a: i64| -?> () { is_float_three32(a) });
    add_primitive!(&mut egraph, "is-float-four32" = |a: i64| -?> () { is_float_four32(a) });
    add_primitive!(&mut egraph, "is-float-half32" = |a: i64| -?> () { is_float_half32(a) });
    add_primitive!(&mut egraph, "is-float-neg-half32" = |a: i64| -?> () { is_float_neg_half32(a) });

    // f32 reciprocal primitives
    add_primitive!(&mut egraph, "has-exact-recip32" = |a: i64| -?> () { has_exact_recip32(a) });
    add_primitive!(
        &mut egraph,
        "float-recip32" = |a: i64| -> i64 { float_recip32(a) }
    );

    // Integer sign primitive
    add_primitive!(&mut egraph, "int-sign" = |a: i64| -> i64 { int_sign(a) });

    // GLSL transcendental constant folding primitives (native f64 via F for FConst)
    add_primitive!(&mut egraph, "float-sin" = |a: F| -> F { F::from(OrderedFloat(a.0.0.sin())) });
    add_primitive!(&mut egraph, "float-cos" = |a: F| -> F { F::from(OrderedFloat(a.0.0.cos())) });
    add_primitive!(&mut egraph, "float-tan" = |a: F| -> F { F::from(OrderedFloat(a.0.0.tan())) });
    add_primitive!(&mut egraph, "float-asin" = |a: F| -> F { F::from(OrderedFloat(a.0.0.asin())) });
    add_primitive!(&mut egraph, "float-acos" = |a: F| -> F { F::from(OrderedFloat(a.0.0.acos())) });
    add_primitive!(&mut egraph, "float-atan" = |a: F| -> F { F::from(OrderedFloat(a.0.0.atan())) });
    add_primitive!(&mut egraph, "float-atan2" = |y: F, x: F| -> F { F::from(OrderedFloat(y.0.0.atan2(x.0.0))) });
    add_primitive!(&mut egraph, "float-sinh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.sinh())) });
    add_primitive!(&mut egraph, "float-cosh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.cosh())) });
    add_primitive!(&mut egraph, "float-tanh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.tanh())) });
    add_primitive!(&mut egraph, "float-asinh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.asinh())) });
    add_primitive!(&mut egraph, "float-acosh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.acosh())) });
    add_primitive!(&mut egraph, "float-atanh" = |a: F| -> F { F::from(OrderedFloat(a.0.0.atanh())) });
    add_primitive!(&mut egraph, "float-exp" = |a: F| -> F { F::from(OrderedFloat(a.0.0.exp())) });
    add_primitive!(&mut egraph, "float-exp2" = |a: F| -> F { F::from(OrderedFloat(a.0.0.exp2())) });
    add_primitive!(&mut egraph, "float-log" = |a: F| -> F { F::from(OrderedFloat(a.0.0.ln())) });
    add_primitive!(&mut egraph, "float-log2" = |a: F| -> F { F::from(OrderedFloat(a.0.0.log2())) });
    add_primitive!(&mut egraph, "float-sqrt" = |a: F| -> F { F::from(OrderedFloat(a.0.0.sqrt())) });
    add_primitive!(&mut egraph, "float-inversesqrt" = |a: F| -> F { F::from(OrderedFloat(1.0 / a.0.0.sqrt())) });
    add_primitive!(&mut egraph, "float-pow" = |x: F, y: F| -> F { F::from(OrderedFloat(x.0.0.powf(y.0.0))) });
    add_primitive!(&mut egraph, "float-floor" = |a: F| -> F { F::from(OrderedFloat(a.0.0.floor())) });
    add_primitive!(&mut egraph, "float-ceil" = |a: F| -> F { F::from(OrderedFloat(a.0.0.ceil())) });
    add_primitive!(&mut egraph, "float-round" = |a: F| -> F { F::from(OrderedFloat(a.0.0.round())) });
    add_primitive!(&mut egraph, "float-trunc" = |a: F| -> F { F::from(OrderedFloat(a.0.0.trunc())) });
    add_primitive!(&mut egraph, "float-abs" = |a: F| -> F { F::from(OrderedFloat(a.0.0.abs())) });
    add_primitive!(&mut egraph, "float-sign" = |a: F| -> F { F::from(OrderedFloat(float_sign_f64(a.0.0))) });
    add_primitive!(&mut egraph, "float-fract" = |a: F| -> F { F::from(OrderedFloat(float_fract_f64(a.0.0))) });
    add_primitive!(&mut egraph, "float-min" = |x: F, y: F| -> F { x.min(y) });
    add_primitive!(&mut egraph, "float-max" = |x: F, y: F| -> F { x.max(y) });
    add_primitive!(&mut egraph, "float-clamp" = |x: F, lo: F, hi: F| -> F { F::from(OrderedFloat(x.0.0.clamp(lo.0.0, hi.0.0))) });
    add_primitive!(&mut egraph, "float-nmin" = |a: F, b: F| -> F { F::from(OrderedFloat(float_nmin(a.0.0, b.0.0))) });
    add_primitive!(&mut egraph, "float-nmax" = |a: F, b: F| -> F { F::from(OrderedFloat(float_nmax(a.0.0, b.0.0))) });
    add_primitive!(&mut egraph, "float-nclamp" = |x: F, lo: F, hi: F| -> F { F::from(OrderedFloat(float_nclamp(x.0.0, lo.0.0, hi.0.0))) });
    add_primitive!(&mut egraph, "float-mix" = |x: F, y: F, a: F| -> F { F::from(OrderedFloat(float_mix_f64(x.0.0, y.0.0, a.0.0))) });
    add_primitive!(&mut egraph, "float-step" = |edge: F, x: F| -> F { F::from(OrderedFloat(if x.0.0 < edge.0.0 { 0.0 } else { 1.0 })) });
    add_primitive!(&mut egraph, "float-smoothstep" = |e0: F, e1: F, x: F| -> F { F::from(OrderedFloat(float_smoothstep_f64(e0.0.0, e1.0.0, x.0.0))) });
    add_primitive!(&mut egraph, "float-rem" = |a: F, b: F| -?> F {
        if b.0.0 == 0.0 { None } else { Some(F::from(OrderedFloat(a.0.0 % b.0.0))) }
    });

    // F-type reciprocal primitives (native f64 for FConst rules)
    add_primitive!(&mut egraph, "f64-has-exact-recip" = |a: F| -?> () {
        f64_has_exact_recip(a.0.0)
    });
    add_primitive!(&mut egraph, "f64-recip" = |a: F| -> F {
        F::from(OrderedFloat(1.0 / a.0.0))
    });

    // QuantizeToF16: round float to IEEE 754 half-precision
    add_primitive!(&mut egraph, "quantize-to-f16" = |a: F| -> F {
        F::from(OrderedFloat(quantize_to_f16(a.0.0)))
    });

    // Type conversion primitives (cross-type: F <-> i64)
    add_primitive!(&mut egraph, "float-to-int-signed" = |a: F| -?> i64 {
        float_to_int_signed(a.0.0)
    });
    add_primitive!(&mut egraph, "float-to-int-unsigned" = |a: F| -?> i64 {
        float_to_int_unsigned(a.0.0)
    });
    add_primitive!(&mut egraph, "int-to-float-signed" = |a: i64| -> F {
        F::from(OrderedFloat(int_to_float_signed(a)))
    });
    add_primitive!(&mut egraph, "int-to-float-unsigned" = |a: i64| -> F {
        F::from(OrderedFloat(int_to_float_unsigned(a)))
    });

    // Unsigned 32-bit comparison primitives (cast to u32 for correct unsigned semantics)
    add_primitive!(&mut egraph, "u32-lt" = |a: i64, b: i64| -?> () { u32_lt(a, b) });
    add_primitive!(&mut egraph, "u32-le" = |a: i64, b: i64| -?> () { u32_le(a, b) });
    add_primitive!(&mut egraph, "u32-gt" = |a: i64, b: i64| -?> () { u32_gt(a, b) });
    add_primitive!(&mut egraph, "u32-ge" = |a: i64, b: i64| -?> () { u32_ge(a, b) });
    add_primitive!(&mut egraph, "u32-min" = |a: i64, b: i64| -> i64 { u32_min(a, b) });
    add_primitive!(&mut egraph, "u32-max" = |a: i64, b: i64| -> i64 { u32_max(a, b) });
    add_primitive!(&mut egraph, "u32-div" = |a: i64, b: i64| -?> i64 { u32_div(a, b) });
    add_primitive!(&mut egraph, "u32-mod" = |a: i64, b: i64| -?> i64 { u32_mod(a, b) });

    // NaN-aware float comparison primitives (FOrd* returns 0 if NaN, FUnord* returns 1 if NaN)
    add_primitive!(&mut egraph, "ford-eq" = |a: F, b: F| -> i64 { ford_eq(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "ford-ne" = |a: F, b: F| -> i64 { ford_ne(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "ford-lt" = |a: F, b: F| -> i64 { ford_lt(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "ford-le" = |a: F, b: F| -> i64 { ford_le(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "ford-gt" = |a: F, b: F| -> i64 { ford_gt(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "ford-ge" = |a: F, b: F| -> i64 { ford_ge(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-eq" = |a: F, b: F| -> i64 { funord_eq(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-ne" = |a: F, b: F| -> i64 { funord_ne(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-lt" = |a: F, b: F| -> i64 { funord_lt(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-le" = |a: F, b: F| -> i64 { funord_le(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-gt" = |a: F, b: F| -> i64 { funord_gt(a.0.0, b.0.0) });
    add_primitive!(&mut egraph, "funord-ge" = |a: F, b: F| -> i64 { funord_ge(a.0.0, b.0.0) });

    // SMod with SPIR-V sign-of-divisor semantics
    add_primitive!(&mut egraph, "smod" = |a: i64, b: i64| -> i64 { smod(a, b) });

    // FMod (floor modulo) primitive for OpFMod constant folding
    add_primitive!(&mut egraph, "float-fmod" = |a: F, b: F| -?> F {
        float_fmod(a.0.0, b.0.0).map(|r| F::from(OrderedFloat(r)))
    });

    // IEEE 754 float negation (sign bit flip, handles ±0.0 correctly)
    add_primitive!(&mut egraph, "float-neg" = |a: F| -> F {
        F::from(OrderedFloat(float_neg(a.0.0)))
    });

    // Safe signed 32-bit division/remainder (guards i32::MIN / -1 overflow)
    add_primitive!(&mut egraph, "sdiv32" = |a: i64, b: i64| -?> i64 { sdiv32(a, b) });
    add_primitive!(&mut egraph, "srem32" = |a: i64, b: i64| -?> i64 { srem32(a, b) });

    // f64 bit pattern predicates for dot product rules
    add_primitive!(&mut egraph, "is-float-one64" = |x: i64| -?> () { is_float_one64(x) });
    add_primitive!(&mut egraph, "is-float-zero64" = |x: i64| -?> () { is_float_zero64(x) });

    // Auto-detecting float multiply: i64 bit patterns → f64 result
    // If both values have zero high 32 bits, treat as f32; otherwise f64
    add_primitive!(&mut egraph, "float-mul-auto" = |a: i64, b: i64| -> F {
        F::from(OrderedFloat(float_mul_auto(a, b)))
    });

    // Auto-detecting float binary ops: i64 bit patterns → i64 bit pattern result
    // Used for vector element-wise constant folding
    add_primitive!(&mut egraph, "float-add-bits" = |a: i64, b: i64| -> i64 {
        float_binop_bits(a, b, std::ops::Add::add, std::ops::Add::add)
    });
    add_primitive!(&mut egraph, "float-sub-bits" = |a: i64, b: i64| -> i64 {
        float_binop_bits(a, b, std::ops::Sub::sub, std::ops::Sub::sub)
    });
    add_primitive!(&mut egraph, "float-mul-bits" = |a: i64, b: i64| -> i64 {
        float_binop_bits(a, b, std::ops::Mul::mul, std::ops::Mul::mul)
    });
    add_primitive!(&mut egraph, "float-div-bits" = |a: i64, b: i64| -> i64 {
        float_binop_bits(a, b, std::ops::Div::div, std::ops::Div::div)
    });
    add_primitive!(&mut egraph, "float-neg-bits" = |a: i64| -> i64 {
        float_neg_bits(a)
    });

    add_primitive!(&mut egraph, "sconvert-fold" = |v: i64, dw: i64| -> i64 {
        sconvert_fold(v, dw)
    });
    add_primitive!(&mut egraph, "uconvert-fold" = |v: i64, sw: i64, dw: i64| -> i64 {
        uconvert_fold(v, sw, dw)
    });

    // Bitcast constant folding primitives (cross int/float reinterpretation)
    add_primitive!(&mut egraph, "bitcast-i32-to-f" = |v: i64| -> F {
        F::from(OrderedFloat(bitcast_int_to_float32(v)))
    });
    add_primitive!(&mut egraph, "bitcast-i64-to-f" = |v: i64| -> F {
        F::from(OrderedFloat(bitcast_int_to_float64(v)))
    });
    add_primitive!(&mut egraph, "bitcast-f-to-i32" = |v: F| -> i64 {
        bitcast_float_to_int32(v.0.0)
    });
    add_primitive!(&mut egraph, "bitcast-f-to-i64" = |v: F| -> i64 {
        bitcast_float_to_int64(v.0.0)
    });

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
mod tests;
