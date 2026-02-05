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

/// Check if an integer, when interpreted as a float, has an exact reciprocal.
/// Returns Some(()) if the float constant has an exact reciprocal representation.
pub fn has_exact_recip(x: i64) -> Option<()> {
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
pub fn float_recip(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    let recip = 1.0 / f;
    recip.to_bits() as i64
}
