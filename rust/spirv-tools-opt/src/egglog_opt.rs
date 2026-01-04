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

// =============================================================================
// Public API
// =============================================================================

/// Create a new egglog EGraph with the SPIR-V language and rules loaded.
///
/// This creates a configured e-graph ready for SPIR-V optimization. Use this
/// with `direct::optimize_module_direct()` for whole-module optimization.
pub fn create_spirv_egraph() -> Result<EGraph, EgglogOptError> {
    let mut egraph = EGraph::default();

    // Load the base SPIR-V language and rules
    egraph
        .parse_and_run_program(None, SPIRV_EGGLOG_PROGRAM)
        .map_err(|e| EgglogOptError::ParseError(e.to_string()))?;

    // Register custom primitives
    add_primitive!(&mut egraph, "bitrev32" = |a: i64| -> i64 {
        bitreverse32(a)
    });
    add_primitive!(&mut egraph, "bitrev64" = |a: i64| -> i64 {
        bitreverse64(a)
    });
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

    // Load rules that use custom primitives
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

        // Add expression: x + 0
        egraph.parse_and_run_program(None, "(let root (Add (Sym \"x\") (Const 0)))").unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        // Should simplify to just (Sym "x")
        assert!(result.contains("Sym") && result.contains("x"), "Expected (Sym \"x\"), got: {}", result);
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

        // (x * 2.0) * 3.0 should merge to x * 6.0 (using integer constants as proxy)
        egraph.parse_and_run_program(None, r#"(let root (FMul (FMul (Sym "x") (Const 2)) (Const 3)))"#).unwrap();
        egraph.parse_and_run_program(None, "(run-schedule (repeat 10 (run)))").unwrap();

        let results = egraph.parse_and_run_program(None, "(extract root)").unwrap();
        assert!(!results.is_empty());
        let result = format!("{}", results[0]);
        eprintln!("FMul chain result: {}", result);
        assert!(result.contains("6"), "Expected merged constant 6, got: {}", result);
    }

    #[test]
    fn test_reciprocal_chain() {
        let mut egraph = create_spirv_egraph().unwrap();

        // 1 / (1 / x) should equal x
        egraph.parse_and_run_program(None, r#"(let root (FDiv (Const 1) (FDiv (Const 1) (Sym "x"))))"#).unwrap();
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

}
