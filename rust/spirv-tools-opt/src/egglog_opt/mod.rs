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
    if roundtrip == f {
        Some(())
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

    // GLSL transcendental constant folding primitives
    add_primitive!(&mut egraph, "float-sin" = |a: i64| -> i64 { float_sin(a) });
    add_primitive!(&mut egraph, "float-cos" = |a: i64| -> i64 { float_cos(a) });
    add_primitive!(&mut egraph, "float-tan" = |a: i64| -> i64 { float_tan(a) });
    add_primitive!(
        &mut egraph,
        "float-asin" = |a: i64| -> i64 { float_asin(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-acos" = |a: i64| -> i64 { float_acos(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-atan" = |a: i64| -> i64 { float_atan(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-atan2" = |y: i64, x: i64| -> i64 { float_atan2(y, x) }
    );
    add_primitive!(
        &mut egraph,
        "float-sinh" = |a: i64| -> i64 { float_sinh(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-cosh" = |a: i64| -> i64 { float_cosh(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-tanh" = |a: i64| -> i64 { float_tanh(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-asinh" = |a: i64| -> i64 { float_asinh(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-acosh" = |a: i64| -> i64 { float_acosh(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-atanh" = |a: i64| -> i64 { float_atanh(a) }
    );
    add_primitive!(&mut egraph, "float-exp" = |a: i64| -> i64 { float_exp(a) });
    add_primitive!(
        &mut egraph,
        "float-exp2" = |a: i64| -> i64 { float_exp2(a) }
    );
    add_primitive!(&mut egraph, "float-log" = |a: i64| -> i64 { float_log(a) });
    add_primitive!(
        &mut egraph,
        "float-log2" = |a: i64| -> i64 { float_log2(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-sqrt" = |a: i64| -> i64 { float_sqrt(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-inversesqrt" = |a: i64| -> i64 { float_inversesqrt(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-pow" = |x: i64, y: i64| -> i64 { float_pow(x, y) }
    );
    add_primitive!(
        &mut egraph,
        "float-floor" = |a: i64| -> i64 { float_floor(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-ceil" = |a: i64| -> i64 { float_ceil(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-round" = |a: i64| -> i64 { float_round(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-trunc" = |a: i64| -> i64 { float_trunc(a) }
    );
    add_primitive!(&mut egraph, "float-abs" = |a: i64| -> i64 { float_abs(a) });
    add_primitive!(
        &mut egraph,
        "float-sign" = |a: i64| -> i64 { float_sign(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-fract" = |a: i64| -> i64 { float_fract(a) }
    );
    add_primitive!(
        &mut egraph,
        "float-min" = |x: i64, y: i64| -> i64 { float_min(x, y) }
    );
    add_primitive!(
        &mut egraph,
        "float-max" = |x: i64, y: i64| -> i64 { float_max(x, y) }
    );
    add_primitive!(
        &mut egraph,
        "float-clamp" = |x: i64, lo: i64, hi: i64| -> i64 { float_clamp(x, lo, hi) }
    );
    add_primitive!(
        &mut egraph,
        "float-mix" = |x: i64, y: i64, a: i64| -> i64 { float_mix(x, y, a) }
    );
    add_primitive!(
        &mut egraph,
        "float-step" = |edge: i64, x: i64| -> i64 { float_step(edge, x) }
    );
    add_primitive!(
        &mut egraph,
        "float-smoothstep" = |e0: i64, e1: i64, x: i64| -> i64 { float_smoothstep(e0, e1, x) }
    );

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
