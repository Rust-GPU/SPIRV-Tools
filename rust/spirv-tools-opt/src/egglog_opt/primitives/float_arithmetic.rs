// =============================================================================
// Float Arithmetic Primitives (for constant folding in FP rules)
// =============================================================================
#![allow(dead_code)]
// These primitives interpret i64 as f32 bit patterns (since SPIRV uses 32-bit floats).
// For 32-bit floats, the bit pattern is stored in the lower 32 bits of i64.

/// Multiply two f32 constants (stored as i64 bit patterns).
pub fn float_mul32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa * fb;
    result.to_bits() as i64
}

/// Divide two f32 constants (stored as i64 bit patterns).
pub fn float_div32(a: i64, b: i64) -> Option<i64> {
    let fb = f32::from_bits(b as u32);
    if fb == 0.0 {
        return None;
    }
    let fa = f32::from_bits(a as u32);
    let result = fa / fb;
    Some(result.to_bits() as i64)
}

/// Add two f32 constants (stored as i64 bit patterns).
pub fn float_add32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa + fb;
    result.to_bits() as i64
}

/// Subtract two f32 constants (stored as i64 bit patterns).
pub fn float_sub32(a: i64, b: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let fb = f32::from_bits(b as u32);
    let result = fa - fb;
    result.to_bits() as i64
}

/// Negate an f32 constant (stored as i64 bit pattern).
pub fn float_neg32(a: i64) -> i64 {
    let fa = f32::from_bits(a as u32);
    let result = -fa;
    result.to_bits() as i64
}

/// Check if an i64 represents f32 1.0 bit pattern.
pub fn is_float_one32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 1.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 0.0 bit pattern.
pub fn is_float_zero32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 -1.0 bit pattern.
pub fn is_float_neg_one32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -1.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 2.0 bit pattern.
pub fn is_float_two32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 2.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 3.0 bit pattern (for powi(3) optimization).
pub fn is_float_three32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 3.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 4.0 bit pattern (for powi(4) optimization).
pub fn is_float_four32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 4.0 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 0.5 bit pattern (for sqrt optimizations).
pub fn is_float_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == 0.5 {
        Some(())
    } else {
        None
    }
}

/// Check if an i64 represents f32 -0.5 bit pattern (for inverse sqrt optimizations).
pub fn is_float_neg_half32(x: i64) -> Option<()> {
    let f = f32::from_bits(x as u32);
    if f == -0.5 {
        Some(())
    } else {
        None
    }
}

/// Check if an f32 constant (stored as i64 bit pattern) has an exact reciprocal.
/// A float has an exact reciprocal iff it is a power of 2 (mantissa bits all zero).
pub fn has_exact_recip32(x: i64) -> Option<()> {
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
pub fn float_recip32(x: i64) -> i64 {
    let f = f32::from_bits(x as u32);
    let recip = 1.0f32 / f;
    recip.to_bits() as i64
}
