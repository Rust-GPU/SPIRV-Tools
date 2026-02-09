// =============================================================================
// Custom Primitives
// =============================================================================
// These functions are called inside `add_primitive!` macro closures in mod.rs,
// which the dead_code lint cannot see through.
#![allow(dead_code)]

/// Reverse the bits of a 32-bit value.
pub fn bitreverse32(x: i64) -> i64 {
    (x as u32).reverse_bits() as i64
}

/// Reverse the bits of a 64-bit value.
pub fn bitreverse64(x: i64) -> i64 {
    (x as u64).reverse_bits() as i64
}

/// Check if two bitmasks are disjoint (no overlapping bits).
pub fn bits_disjoint(a: i64, b: i64) -> Option<()> {
    if (a & b) == 0 {
        Some(())
    } else {
        None
    }
}

/// Check if left shift would clear all bits in a mask.
pub fn shl_clears_mask(mask: i64, shift: i64) -> Option<()> {
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
pub fn shr_clears_mask(mask: i64, shift: i64) -> Option<()> {
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
pub fn mask_superset(a: i64, b: i64) -> Option<()> {
    if (a & b) == b {
        Some(())
    } else {
        None
    }
}

/// Find the least significant bit position (0-indexed), or -1 if zero.
pub fn find_lsb(x: i64) -> i64 {
    if x == 0 {
        -1
    } else {
        (x as u64).trailing_zeros() as i64
    }
}

/// Find the most significant bit position for unsigned (0-indexed), or -1 if zero.
pub fn find_msb_unsigned(x: i64) -> i64 {
    if x == 0 {
        -1
    } else {
        63 - (x as u64).leading_zeros() as i64
    }
}

/// Find the most significant bit position for signed values.
/// For positive: position of highest 1-bit
/// For negative: position of highest 0-bit
pub fn find_msb_signed(x: i64) -> i64 {
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
pub fn popcount(x: i64) -> i64 {
    (x as u64).count_ones() as i64
}

/// Check if a value is a power of 2 (and positive).
/// Returns Some(()) if x is a power of 2, None otherwise.
pub fn is_pow2(x: i64) -> Option<()> {
    if x > 0 && (x & (x - 1)) == 0 {
        Some(())
    } else {
        None
    }
}

/// Compute log base 2 of a power of 2.
/// Assumes x is a positive power of 2 (caller should guard with is-pow2).
pub fn log2_pow2(x: i64) -> i64 {
    if x <= 0 {
        return -1;
    }
    (x as u64).trailing_zeros() as i64
}

/// Check if an integer, when interpreted as a f64, has an exact reciprocal.
/// A float has an exact reciprocal iff it is a power of 2 (mantissa bits all zero).
pub fn has_exact_recip(x: i64) -> Option<()> {
    let bits = x as u64;
    let f = f64::from_bits(bits);
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    // f64 mantissa is 52 bits. A power of 2 has all mantissa bits zero.
    const F64_MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    if (bits & F64_MANTISSA_MASK) == 0 {
        // Also check the reciprocal is finite (excludes extreme exponents)
        let recip = 1.0 / f;
        if recip.is_finite() {
            Some(())
        } else {
            None
        }
    } else {
        None
    }
}

/// Compute the reciprocal of a float constant (stored as i64 bits).
pub fn float_recip(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    let recip = 1.0 / f;
    recip.to_bits() as i64
}

/// Check if a native f64 has an exact reciprocal (power of 2 check).
/// Used for F-type primitives operating on FConst values.
pub fn f64_has_exact_recip(f: f64) -> Option<()> {
    if !f.is_finite() || f == 0.0 {
        return None;
    }
    const F64_MANTISSA_MASK: u64 = (1u64 << 52) - 1;
    let bits = f.to_bits();
    if (bits & F64_MANTISSA_MASK) == 0 {
        let recip = 1.0 / f;
        if recip.is_finite() {
            Some(())
        } else {
            None
        }
    } else {
        None
    }
}

// =============================================================================
// Unsigned 32-bit comparison/arithmetic primitives
// =============================================================================
// Constants are stored as sign-extended i64, so 0xFFFFFFFF becomes -1.
// These primitives cast to u32 for correct unsigned semantics.

pub fn u32_lt(a: i64, b: i64) -> Option<()> {
    if (a as u32) < (b as u32) {
        Some(())
    } else {
        None
    }
}

pub fn u32_le(a: i64, b: i64) -> Option<()> {
    if (a as u32) <= (b as u32) {
        Some(())
    } else {
        None
    }
}

pub fn u32_gt(a: i64, b: i64) -> Option<()> {
    if (a as u32) > (b as u32) {
        Some(())
    } else {
        None
    }
}

pub fn u32_ge(a: i64, b: i64) -> Option<()> {
    if (a as u32) >= (b as u32) {
        Some(())
    } else {
        None
    }
}

pub fn u32_min(a: i64, b: i64) -> i64 {
    (a as u32).min(b as u32) as i32 as i64
}

pub fn u32_max(a: i64, b: i64) -> i64 {
    (a as u32).max(b as u32) as i32 as i64
}

pub fn u32_div(a: i64, b: i64) -> Option<i64> {
    let b = b as u32;
    if b == 0 {
        None
    } else {
        Some((a as u32 / b) as i32 as i64)
    }
}

pub fn u32_mod(a: i64, b: i64) -> Option<i64> {
    let b = b as u32;
    if b == 0 {
        None
    } else {
        Some((a as u32 % b) as i32 as i64)
    }
}

// =============================================================================
// Type conversion primitives (cross-type: F <-> i64)
// =============================================================================

/// Convert f64 to signed i32, sign-extended to i64. Returns None for NaN/Inf/out-of-range.
pub fn float_to_int_signed(f: f64) -> Option<i64> {
    if !f.is_finite() {
        return None;
    }
    let truncated = f as i64; // Rust saturates, so check range
    if truncated < i32::MIN as i64 || truncated > i32::MAX as i64 {
        return None;
    }
    Some(truncated)
}

/// Convert f64 to unsigned u32, sign-extended to i64. Returns None for NaN/Inf/negative/out-of-range.
pub fn float_to_int_unsigned(f: f64) -> Option<i64> {
    if !f.is_finite() || f < 0.0 {
        return None;
    }
    let truncated = f as u64; // Rust saturates
    if truncated > u32::MAX as u64 {
        return None;
    }
    // Sign-extend u32 to i64 (matching how constants are stored)
    Some(truncated as u32 as i32 as i64)
}

/// Convert signed i32 (stored as sign-extended i64) to f64.
pub fn int_to_float_signed(x: i64) -> f64 {
    (x as i32) as f64
}

/// Convert unsigned u32 (stored as sign-extended i64) to f64.
pub fn int_to_float_unsigned(x: i64) -> f64 {
    (x as u32) as f64
}

// =============================================================================
// NaN-aware float comparison primitives
// =============================================================================
// IEEE 754: FOrd* returns false if either operand is NaN.
// IEEE 754: FUnord* returns true if either operand is NaN.
// egglog uses OrderedFloat where NaN==NaN is true, so we need custom primitives.

pub fn ford_eq(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a == b {
        1
    } else {
        0
    }
}
pub fn ford_ne(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a != b {
        1
    } else {
        0
    }
}
pub fn ford_lt(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a < b {
        1
    } else {
        0
    }
}
pub fn ford_le(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a <= b {
        1
    } else {
        0
    }
}
pub fn ford_gt(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a > b {
        1
    } else {
        0
    }
}
pub fn ford_ge(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() {
        0
    } else if a >= b {
        1
    } else {
        0
    }
}
pub fn funord_eq(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a == b {
        1
    } else {
        0
    }
}
pub fn funord_ne(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a != b {
        1
    } else {
        0
    }
}
pub fn funord_lt(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a < b {
        1
    } else {
        0
    }
}
pub fn funord_le(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a <= b {
        1
    } else {
        0
    }
}
pub fn funord_gt(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a > b {
        1
    } else {
        0
    }
}
pub fn funord_ge(a: f64, b: f64) -> i64 {
    if a.is_nan() || b.is_nan() || a >= b {
        1
    } else {
        0
    }
}

// =============================================================================
// SMod with SPIR-V sign-of-divisor semantics
// =============================================================================

/// SPIR-V SMod: result has the same sign as the divisor.
/// result = a - b * floor(a/b), or equivalently: r = a%b; if sign differs, r += b.
/// Returns 0 for division by zero (matching C++ parity).
pub fn smod(a: i64, b: i64) -> i64 {
    if b == 0 {
        return 0;
    }
    let (a32, b32) = (a as i32, b as i32);
    // Use wrapping_rem to avoid panic on i32::MIN % -1
    let mut result = a32.wrapping_rem(b32);
    if result != 0 && (b32 < 0) != (result < 0) {
        result = result.wrapping_add(b32);
    }
    result as i64
}

// =============================================================================
// FMod (floor modulo) primitive
// =============================================================================

/// SPIR-V OpFMod: result = x - y * floor(x/y).
/// Different from Rust's % (which is truncated remainder = OpFRem).
pub fn float_fmod(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 || a.is_nan() || b.is_nan() || a.is_infinite() {
        return None;
    }
    Some(a - b * (a / b).floor())
}

// =============================================================================
// f64 float bit pattern predicates
// =============================================================================

/// IEEE 754 float negation: flip the sign bit.
/// Unlike `0.0 - a`, this correctly handles: neg(+0.0) = -0.0, neg(-0.0) = +0.0.
pub fn float_neg(a: f64) -> f64 {
    -a
}

/// Safe signed 32-bit division. Returns None for i32::MIN / -1 (overflow)
/// and for division by zero. This is safer than C++ which silently produces
/// undefined results for these cases.
pub fn sdiv32(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let (a32, b32) = (a as i32, b as i32);
    // i32::MIN / -1 overflows — don't fold (UB in SPIR-V)
    if a32 == i32::MIN && b32 == -1 {
        return None;
    }
    Some((a32 / b32) as i64)
}

/// Safe signed 32-bit remainder. Returns None for i32::MIN % -1 (overflow)
/// and for division by zero.
pub fn srem32(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let (a32, b32) = (a as i32, b as i32);
    if a32 == i32::MIN && b32 == -1 {
        return None;
    }
    Some((a32 % b32) as i64)
}

/// Check if an i64, interpreted as f64 bit pattern, equals 1.0.
pub fn is_float_one64(x: i64) -> Option<()> {
    let f = f64::from_bits(x as u64);
    if f == 1.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64, interpreted as f64 bit pattern, equals +0.0.
pub fn is_float_zero64(x: i64) -> Option<()> {
    let f = f64::from_bits(x as u64);
    if f == 0.0 {
        Some(())
    } else {
        None
    }
}
