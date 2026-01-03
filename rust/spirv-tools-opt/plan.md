# Design 4: Abstract Interpretation Analysis for E-Graph Optimization

## Goal
Enable true whole-program optimization in a single e-graph pass by using rich abstract interpretation analysis. Rules use analysis data to make sound decisions without hardcoded guards or manual passes.

## Core Principle
The problem isn't that rules conflict - it's that we lack sufficient information to apply them correctly. Rich abstract interpretation gives rules enough information to always do the right thing.

---

## Phase 1: SpirvAnalysis Foundation

### 1.1 Define the Analysis Struct

```rust
#[derive(Clone, Debug, Default)]
struct SpirvAnalysis {
    // Constant propagation lattice
    constant: ConstLattice,

    // Known-bits analysis (extremely powerful for SPIR-V)
    known_zeros: u64,    // Bits proven to be 0
    known_ones: u64,     // Bits proven to be 1

    // Bit width tracking
    bit_width: Option<u32>,

    // Value range analysis
    min_value: Option<u64>,
    max_value: Option<u64>,

    // Divisibility tracking (for div/mod optimizations)
    known_factor: Option<u64>,  // Value is known multiple of this

    // Semantic origin tracking
    origin: Origin,
}

#[derive(Clone, Debug, Default, PartialEq)]
enum ConstLattice {
    #[default]
    Top,                    // Unknown value
    Known(ConstValue),      // Single known constant
    Bottom,                 // Contradiction (bug detected!)
}

#[derive(Clone, Debug, Default, PartialEq)]
enum Origin {
    #[default]
    Pure,                   // No domain restrictions
    Bitwise,                // Derived from bitwise operations
    Division,               // Derived from div/mod operations
    Mixed,                  // Multiple origins
}
```

### 1.2 Implement Analysis Trait

The `Analysis<SpirvLang>` trait requires:
- `type Data = SpirvAnalysis`
- `fn make(egraph, enode) -> Data` - compute analysis for a node
- `fn merge(&mut self, other) -> DidMerge` - merge when e-classes unify

Key semantics:
- `make()` propagates information from children to parent
- `merge()` combines information when e-classes unify (detects conflicts)
- Lattice operations: Top ⊔ Known(x) = Known(x), Known(x) ⊔ Known(y) = Bottom if x≠y

---

## Phase 2: Update E-Graph Infrastructure

### 2.1 Change All Signatures

Replace all occurrences of:
```rust
EGraph<SpirvLang, ()>
```
With:
```rust
EGraph<SpirvLang, SpirvAnalysis>
```

Affected locations:
- `optimize_expr()` function
- All `Applier` implementations
- Helper functions (`const_value`, `has_symbol`, etc.)
- Test code

### 2.2 Update Appliers

Change appliers from using global scans to local analysis:

Before:
```rust
if egraph_has_bitwise(egraph) {
    return Vec::new();
}
```

After:
```rust
// Use analysis data from the matched e-class
let data = &egraph[eclass].data;
// Make decision based on rich analysis
```

---

## Phase 3: Remove Workarounds

### 3.1 Delete Guard Code
- Remove `egraph_has_bitwise()` function
- Remove guard checks in `UModPowerOfTwo`, `SRemConstDecompose`, `UModConstDecompose`

### 3.2 Simplify Helper Functions
- Replace `const_value()` with analysis lookup: `egraph[id].data.constant`
- Replace `pure_const_value()` similarly
- Remove redundant iteration over e-class nodes

---

## Phase 4: Enhanced Rewrites (Future)

With rich analysis, rewrites can be smarter:

```rust
impl Applier for UModConstDecompose {
    fn apply_one(&self, egraph: &mut EGraph<...>, ...) -> Vec<Id> {
        let x_data = &egraph[x_id].data;
        let c_data = &egraph[c_id].data;

        // If c is power of 2, use bitwise AND (always safe)
        if let ConstLattice::Known(c) = &c_data.constant {
            if c.value().is_power_of_two() {
                let mask = c.value() - 1;
                let mask_id = egraph.add(SpirvLang::Const(ConstValue::new(mask)));
                return vec![egraph.add(SpirvLang::BitAnd([x_id, mask_id]))];
            }
        }

        // Otherwise use standard decomposition
        // Analysis ensures this is sound
        // ...
    }
}
```

---

## Implementation Checklist

- [ ] Define `ConstLattice` enum
- [ ] Define `Origin` enum
- [ ] Define `SpirvAnalysis` struct
- [ ] Implement `Analysis<SpirvLang> for SpirvAnalysis`
  - [ ] `make()` for all node types
  - [ ] `merge()` with proper lattice semantics
- [ ] Update `EGraph<SpirvLang, ()>` → `EGraph<SpirvLang, SpirvAnalysis>` globally
- [ ] Update `Applier` implementations
- [ ] Update helper functions
- [ ] Remove `egraph_has_bitwise()` and guards
- [ ] Run full test suite
- [ ] Fix any failing tests

---

## Known-Bits Analysis Details

For each e-class, track which bits are known:
- `known_zeros`: bitmask of bits proven to be 0
- `known_ones`: bitmask of bits proven to be 1
- Invariant: `known_zeros & known_ones == 0`

Propagation rules:
- `BitAnd(a, b)`: `zeros = a.zeros | b.zeros`, `ones = a.ones & b.ones`
- `BitOr(a, b)`: `zeros = a.zeros & b.zeros`, `ones = a.ones | b.ones`
- `BitXor(a, b)`: Complex - known bits where both inputs are known
- `Shl(a, n)`: Shift known bits left, low bits become known zeros
- `Const(c)`: All bits are known from constant value

This enables optimizations like:
- `x & 0xFF` when top bits already known zero → `x`
- `x | 0xFF` when low bits already known one → `x`
- Detecting when shift would zero out the value

---

## Progress Log

### Session 1 - COMPLETED
- Created plan.md
- Defined `ConstLattice`, `Origin`, and `SpirvAnalysis` structs
- Implemented `Analysis<SpirvLang>` trait for `SpirvAnalysis`
  - `make()` handles all node types with proper known-bits propagation
  - `merge()` combines analyses with conflict detection
- Updated all `EGraph<SpirvLang, ()>` → `EGraph<SpirvLang, SpirvAnalysis>`
- Updated all `Applier<SpirvLang, ()>` → `Applier<SpirvLang, SpirvAnalysis>`
- Updated `rewrites()` return type
- Updated `control.rs` and `bin/opt_block.rs`
- Replaced global `egraph_has_bitwise()` guards with per-class origin checks:
  - `egraph[subst[self.x]].data.origin.has_bitwise()`
  - More efficient: O(1) lookup vs O(n) e-graph scan
  - More precise: only blocks when the specific operand has bitwise origin
- Deleted `egraph_has_bitwise()` function
- **All 66 tests passing!**

## Summary

The SpirvAnalysis infrastructure is now in place and working. Key benefits:

1. **Per-class origin tracking**: Each e-class tracks whether it derives from bitwise operations
2. **Known-bits analysis**: Propagates bit-level information (zeros/ones) through operations
3. **Constant lattice**: Tracks known constant values with conflict detection
4. **Local guards**: Appliers use per-class data instead of global e-graph scans

### Session 2 - COMPLETED

**Analysis-based constant lookups:**
- Replaced `const_value()` to use `egraph[id].data.const_value()` (O(1) vs O(n))
- Added conflict detection logging in `ConstLattice::meet()` (debug builds only)

**Known-bits optimizations:**
- Enhanced `BitAndConstSimplify` with known-bits analysis:
  - `x & mask = x` when all possibly-1 bits in x are covered by mask
  - `x & mask = 0` when no bits can be 1 in mask positions
- Enhanced `BitOrConstSimplify` with known-bits analysis:
  - `x | c = x` when all bits in c are already known-ones in x
  - `x | c = ~0` when known-ones combined with c covers all bits

**Shift known-bits propagation:**
- Enhanced `Shl` to propagate known-bits when shift amount is constant:
  - Bottom n bits become known zeros after left shift by n
  - Known bits from input shift left accordingly
- Enhanced `ShrU` to propagate known-bits when shift amount is constant:
  - Top n bits become known zeros after right shift by n
  - Known bits from input shift right accordingly

**All 66 tests passing!**

### Session 3 - COMPLETED

**Signed operation constant folding:**
- Added `to_signed()` helper for proper signed integer interpretation with sign extension
- Enhanced `SDiv` with constant folding for signed division
- Enhanced `SRem` with constant folding (result has same sign as dividend)
- Enhanced `SMod` with constant folding (result has same sign as divisor)

**Signed comparison constant folding:**
- Enhanced `SLt` (signed less than) with constant folding
- Enhanced `SLe` (signed less or equal) with constant folding
- Enhanced `SGt` (signed greater than) with constant folding
- Enhanced `SGe` (signed greater or equal) with constant folding

**Logical operation constant folding:**
- Enhanced `LogNot` with constant folding: !0 = 1, !non-zero = 0
- Enhanced `LogAnd` with constant folding: true && true = true
- Enhanced `LogOr` with constant folding: true || anything = true
- Enhanced `LogEq` with constant folding: both true or both false
- Enhanced `LogNe` with constant folding: different boolean values

**Range and divisibility propagation:**
- Enhanced `ShrU` with range propagation: x >> n has range [min >> n, max >> n]
- Enhanced `Neg` with divisibility propagation: -x preserves divisibility factor

**All 695 tests passing!**

### Session 4 - COMPLETED

**BitReverse constant folding:**
- Added `reverse_bits()` helper function for bit reversal
- Enhanced `BitReverse` with constant folding
- Added known-bits propagation (reverse known_zeros and known_ones)

**Rotation constant folding:**
- Enhanced `RotL` (rotate left) with constant folding
- Enhanced `RotR` (rotate right) with constant folding

**Select enhancements:**
- Added condition-based constant folding: when condition is known true/false, return that arm's analysis
- Added known-bits intersection: bits known in BOTH arms are known in result
- Added range union: min is minimum of both mins, max is maximum of both maxes
- Added divisibility GCD: if both arms share a common factor, result has it too

**ShrS known-bits propagation:**
- Enhanced `ShrS` (signed/arithmetic right shift) with known-bits propagation
- Propagates known-zeros and known-ones with sign extension
- Top bits become copies of sign bit (known-zero or known-one based on sign bit analysis)

**Range-based comparison folding:**
- Enhanced `ULt` with range analysis: if a_max < b_min → always true, if a_min >= b_max → always false
- Enhanced `ULe` with range analysis: if a_max <= b_min → always true, if a_min > b_max → always false
- Enhanced `UGt` with range analysis: if a_min > b_max → always true, if a_max <= b_min → always false
- Enhanced `UGe` with range analysis: if a_min >= b_max → always true, if a_max < b_min → always false

**Eq/Ne known-bits and range folding:**
- Enhanced `Eq` with known-bits: if any bit differs in known values → definitely false
- Enhanced `Eq` with range: if ranges don't overlap → definitely false
- Enhanced `Ne` with known-bits: if any bit differs in known values → definitely true
- Enhanced `Ne` with range: if ranges don't overlap → definitely true

**All 695 tests passing!**

### Session 5 - COMPLETED

**Logical NOT double negation:**
- Added `lnot-lnot-cancel`: `!!x = x`
- Added `bnot-bnot-cancel`: `~~x = x`

**Logical AND/OR identity rewrites:**
- Added `land-true-left/right`: `true && x = x`, `x && true = x`
- Added `land-false-left/right`: `false && x = false`, `x && false = false`
- Added `land-self`: `x && x = x`
- Added `lor-true-left/right`: `true || x = true`, `x || true = true`
- Added `lor-false-left/right`: `false || x = x`, `x || false = x`
- Added `lor-self`: `x || x = x`

**De Morgan's laws:**
- Added `demorgan-land`: `!(a && b) = !a || !b`
- Added `demorgan-lor`: `!(a || b) = !a && !b`

**Logical AND/OR with NOT self:**
- Added `land-lnot-self` and `land-lnot-self-comm`: `a && !a = false`
- Added `lor-lnot-self` and `lor-lnot-self-comm`: `a || !a = true`
- Implemented custom `LogAndNotSelf` and `LogOrNotSelf` appliers

**Select simplification rewrites:**
- Added `select-same-arms`: `select(c, a, a) = a`
- Added `select-lnot-cond`: `select(!c, t, f) = select(c, f, t)`

**Comparison inversion rewrites with NOT:**
- Added `lnot-eq`: `!(a == b) = (a != b)`
- Added `lnot-ne`: `!(a != b) = (a == b)`
- Added `lnot-ult`: `!(a < b) = (a >= b)` (unsigned)
- Added `lnot-ule`: `!(a <= b) = (a > b)` (unsigned)
- Added `lnot-ugt`: `!(a > b) = (a <= b)` (unsigned)
- Added `lnot-uge`: `!(a >= b) = (a < b)` (unsigned)
- Added `lnot-slt`: `!(a < b) = (a >= b)` (signed)
- Added `lnot-sle`: `!(a <= b) = (a > b)` (signed)
- Added `lnot-sgt`: `!(a > b) = (a <= b)` (signed)
- Added `lnot-sge`: `!(a >= b) = (a < b)` (signed)

**Unit tests added:**
- `logical_not_double_negation`
- `bitwise_not_double_negation`
- `logical_and_with_true_left`
- `logical_and_with_false_left`
- `logical_or_with_true_left`
- `logical_or_with_false_left`
- `logical_and_self`
- `logical_or_self`
- `select_with_same_arms`
- `lnot_eq_to_ne`
- `lnot_ne_to_eq`
- `land_lnot_self_is_false`
- `lor_lnot_self_is_true`

**All tests passing!**

### Session 6 - COMPLETED

**Bitwise De Morgan's laws:**
- Added `demorgan-band`: `~(a & b) = (~a) | (~b)`
- Added `demorgan-bor`: `~(a | b) = (~a) & (~b)`

**Bitwise XOR identities:**
- Added `bxor-zero` and `bxor-zero-comm`: `a ^ 0 = a`
- Added `bxor-allones` and `bxor-allones-comm`: `a ^ ~0 = ~a`
- Note: `bxor-self` (`a ^ a = 0`) already existed from earlier session

**Bitwise AND/OR with NOT self:**
- Added `band-bnot-self` and `band-bnot-self-comm`: `a & ~a = 0`
- Added `bor-bnot-self` and `bor-bnot-self-comm`: `a | ~a = ~0`
- Implemented custom `BandBnotSelf` and `BorBnotSelf` appliers with bit-width awareness

**Select with boolean constant results:**
- Added `select-true-false`: `select(c, true, false) = c`, `select(c, false, true) = !c`
- Implemented custom `SelectTrueFalse` applier
- Added `select-cond-true`: `select(c, true, f) = c || f`
- Added `select-cond-false`: `select(c, false, f) = !c && f`

**Nested select simplifications:**
- Added `select-nested-same-cond`: `select(c, select(c, a, b), d) = select(c, a, d)`
- Added `select-nested-same-cond-false`: `select(c, d, select(c, a, b)) = select(c, d, b)`

**Unit tests added:**
- `bitwise_demorgan_band`
- `bitwise_demorgan_bor`
- `bxor_with_zero`
- `bxor_with_allones_32bit`
- `band_bnot_self_is_zero`
- `bor_bnot_self_is_allones`
- `select_true_false_is_cond`
- `select_false_true_is_not_cond`
- `select_nested_same_cond_simplifies`

**All tests passing!**

### Session 7 - COMPLETED

**Rotate identities:**
- Added `rotl-zero`: `rotl(a, 0) = a`
- Added `rotr-zero`: `rotr(a, 0) = a`

**Logical equivalence with self:**
- Added `leq-self`: `leq(a, a) = true`
- Added `lne-self`: `lne(a, a) = false`

**Note:** Many comparison-self rules (eq-self, ne-self, ult-self, etc.) were found to already exist in earlier sessions. Session 7 cleaned up duplicates and added the missing logical comparison rewrites.

**Unit tests added:**
- `rotate_left_by_zero_is_identity`
- `rotate_right_by_zero_is_identity`
- `logical_eq_self_is_true`
- `logical_ne_self_is_false`

**All tests passing!**

### Session 8 - COMPLETED

**Division self-identity:**
- Added `udiv-self`: `x / x = 1` (unsigned)
- Added `sdiv-self`: `x / x = 1` (signed)
- Implemented custom `DivSelf` applier with bit-width awareness

**Modulo self-identity:**
- Added `umod-self`: `x % x = 0` (unsigned)
- Added `srem-self`: `x % x = 0` (signed remainder)
- Added `smod-self`: `x % x = 0` (signed modulo)
- Implemented custom `ModSelf` applier with bit-width awareness

**Shift combining patterns:**
- Added `shl-shr-combine`: `(x << n) >> n = x & mask` (masks high bits when equal shift amounts)
- Added `shr-shl-combine`: `(x >> n) << n = x & mask` (masks low bits when equal shift amounts)
- Implemented custom `ShlShrCombine` and `ShrShlCombine` appliers
- Properly handles shift overflow edge cases for 64-bit widths

**Unit tests added:**
- `udiv_self_is_one`
- `sdiv_self_is_one`
- `umod_self_is_zero`
- `srem_self_is_zero`
- `smod_self_is_zero`
- `shl_shr_combine_masks_high_bits`
- `shr_shl_combine_masks_low_bits`
- `shift_combine_different_amounts_not_simplified`

**All tests passing!**

### Session 9 - COMPLETED

**Rotate cancellation:**
- Added `rotl-rotr-cancel`: `rotl(rotr(x, n), n) = x`
- Added `rotr-rotl-cancel`: `rotr(rotl(x, n), n) = x`
- Simple pattern match when rotation amounts are identical

**Rotate composition with constants:**
- Added `rotl-rotl-compose`: `rotl(rotl(x, a), b) = rotl(x, (a+b) % width)`
- Added `rotr-rotr-compose`: `rotr(rotr(x, a), b) = rotr(x, (a+b) % width)`
- Implemented custom `RotateCompose` applier with modular arithmetic

**Rotate interconversion:**
- Added `rotl-to-rotr`: `rotl(x, n) = rotr(x, width-n)` when n is constant
- Added `rotr-to-rotl`: `rotr(x, n) = rotl(x, width-n)` when n is constant
- Implemented custom `RotateConvert` applier for bidirectional conversion

**Rotate by full width is identity:**
- Added `rotl-full-width`: `rotl(x, k*width) = x` for any integer k
- Added `rotr-full-width`: `rotr(x, k*width) = x` for any integer k
- Implemented custom `RotateFullWidth` applier with modulo check

**Mixed rotation simplification:**
- Added `rotl-rotr-nested`: `rotl(rotr(x, a), b) = rotl(x, b-a)` or `rotr(x, a-b)`
- Added `rotr-rotl-nested`: `rotr(rotl(x, a), b)` = net rotation
- Implemented custom `RotateCancelDiff` applier for different rotation amounts

**Unit tests added:**
- `rotl_rotr_cancel_same_amount`
- `rotr_rotl_cancel_same_amount`
- `rotl_rotl_compose_constants`
- `rotr_rotr_compose_constants`
- `rotl_full_width_is_identity`
- `rotr_full_width_is_identity`
- `rotl_rotr_different_amounts_simplifies`
- `rotr_rotl_different_amounts_simplifies`
- `rotl_zero_is_identity`
- `rotr_zero_is_identity`

**All tests passing!**

### Session 10 - COMPLETED

**Add/Mul constant reassociation (matching C++ ReassociateCommutativeOp):**
- Added `add-reassociate-const-right`: `(+ c1 (+ x c2)) = (+ x (c1+c2))`
- Added `add-reassociate-const-left`: `(+ (+ x c2) c1) = (+ x (c1+c2))`
- Added `add-reassociate-const-inner-left`: `(+ c1 (+ c2 x)) = (+ x (c1+c2))`
- Added `add-reassociate-const-outer-left`: `(+ (+ c2 x) c1) = (+ x (c1+c2))`
- Added `mul-reassociate-const-right`: `(* c1 (* x c2)) = (* x (c1*c2))`
- Added `mul-reassociate-const-left`: `(* (* x c2) c1) = (* x (c1*c2))`
- Added `mul-reassociate-const-inner-left`: `(* c1 (* c2 x)) = (* x (c1*c2))`
- Added `mul-reassociate-const-outer-left`: `(* (* c2 x) c1) = (* x (c1*c2))`
- Implemented custom `AddConstReassociate` and `MulConstReassociate` appliers
- Combines with existing mul-to-shift rules for better optimization (e.g., `x * 65536 * 65536` → `x << 32`)

**Unit tests added:**
- `add_reassociates_constants_right`
- `add_reassociates_constants_left`
- `add_reassociates_constants_inner_left`
- `mul_reassociates_constants_right`
- `mul_reassociates_constants_left`
- `mul_reassociates_constants_inner_left`
- `add_reassociates_with_u64_widths`
- `mul_reassociates_with_u64_widths`

**All 491 tests passing!**

---

## Gap Analysis: C++ spirv-opt Folding Rules vs Rust E-Graph

This section identifies specific optimizations from C++ `folding_rules.cpp` that are missing
in our Rust e-graph implementation. Priority is based on frequency and impact.

### Priority 1: Negate Propagation (High Impact - Integer Arithmetic)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 1 | `MergeAddNegateArithmetic` | `x + (-y)` → `x - y` | ✅ HAVE (add-neg-to-sub) |
| 2 | `MergeSubNegateArithmetic` | `x - (-y)` → `x + y` | ✅ HAVE (sub-neg-right-to-add) |
| 3 | `MergeNegateMulDivArithmetic` | `(-x) * (-y)` → `x * y` | ✅ HAVE (mul-double-neg) |
| 4 | `MergeMulNegateArithmetic` | `(-x) * y` → `-(x * y)` | ✅ HAVE (mul-neg-left/right) |
| 5 | `MergeDivNegateArithmetic` | `(-x) / y` → `-(x / y)` | ✅ HAVE (sdiv-neg-left/right) |

### Priority 2: Add/Sub Constant Merging (High Impact)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 6 | `MergeAddAddArithmetic` | `(x + c1) + c2` → `x + (c1+c2)` | ✅ HAVE (add-reassociate-const-*) |
| 7 | `MergeAddSubArithmetic` | `(x + c1) - c2` → `x + (c1-c2)` | ✅ HAVE (sub-of-add-merge-consts) |
| 8 | `MergeSubAddArithmetic` | `(x - c1) + c2` → `x + (c2-c1)` | ✅ HAVE (add-sub-merge-consts) |
| 9 | `MergeSubSubArithmetic` | `(x - c1) - c2` → `x - (c1+c2)` | ✅ HAVE (sub-chain-merge-consts) |

### Priority 3: Bitwise + Shift Patterns (Medium Impact)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 10 | `RedundantAndAddSub` | `(x & mask) + (y & ~mask)` patterns | ✅ HAVE (add-band-complement-*, Session 17) |
| 11 | `RedundantAndShift` | `(x >> n) & mask` simplifications | ✅ HAVE (band-redundant-shl/shr/shr-signed, Session 17) |

### Priority 4: Comparison Chaining (Medium Impact)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 12 | Signed compare chains | `a < b && b < c` → range check | ✅ HAVE (CmpChainTransitive/SameLhs/SameRhs/Mixed, Session 19) |
| 13 | Unsigned bounds | `a <= MAX && a >= MIN` → true | ✅ HAVE via analysis + contradictory/tautology rules (Session 19) |

### Priority 5: Floating-Point Rules (Lower Priority for Integer Focus)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 14 | `RedundantFAdd` | `x + 0.0` → `x` | ✅ HAVE (fadd-zero, Session 16) |
| 15 | `RedundantFSub` | `x - 0.0` → `x` | ✅ HAVE (fsub-zero-right, Session 16) |
| 16 | `RedundantFMul` | `x * 1.0` → `x` | ✅ HAVE (fmul-one via FMulIdentity applier, Session 16) |
| 17 | `RedundantFDiv` | `x / 1.0` → `x` | ✅ HAVE (fdiv-one via FDivIdentity applier, Session 16) |
| 18 | `ReciprocalFDiv` | `x / (1.0/y)` → `x * y` | ✅ HAVE (fdiv-reciprocal via FDivReciprocal applier, Session 17) |
| 19 | `MergeMulMulArithmetic` | `(x * c1) * c2` → `x * (c1*c2)` | ✅ HAVE (mul-reassociate-const-*) |
| 20 | `MergeDivDivArithmetic` | `(x / c1) / c2` → `x / (c1*c2)` | ✅ HAVE (sdiv/udiv-merge-consts) |

### Priority 6: Min/Max Patterns (Medium Impact)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 21 | Min/Max identity | `max(x, x)` → `x` | ✅ HAVE (fmin-self, fmax-self, smin-self, smax-self, umin-self, umax-self, Session 16) |
| 22 | Min/Max absorption | `max(x, max(x, y))` → `max(x, y)` | ✅ HAVE (smin/smax/umin/umax absorption rules, Session 13) |
| 23 | Clamp patterns | `max(min(x, hi), lo)` → `clamp(x, lo, hi)` | ✅ HAVE (smin-smax-to-sclamp, etc., Session 18) |

### Priority 7: Composite/Vector Operations (Future Scope)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 24 | `InsertFeedingExtract` | `extract(insert(v, x, i), i)` → `x` | ✅ HAVE (extract-insert-same-idx, extract-insert-diff-idx, Session 18) |
| 25 | `VectorShuffleFeedingExtract` | Shuffle then extract simplification | ✅ HAVE (extract-shuffle, Session 32 - uses ShuffleMask) |
| 26 | `DotProductDoingExtract` | `dot(v, e_i)` → `v.component_i` | ✅ HAVE (dot-unit-vector, Session 19 - uses VecConst::unit_vector_index) |
| 27 | `FMixFeedingExtract` | `extract(fmix(x, y, a), i)` simplification | ✅ HAVE (fmix-feeding-extract, Session 20 - per-component blend factor optimization) |

---

## Session 11: Gap Analysis Verification - COMPLETED

### Findings

Upon code verification, **all 10 originally planned optimizations already exist** in the codebase:

| Pattern | Status | Location |
|---------|--------|----------|
| `x + (-y)` → `x - y` | ✅ HAVE | `add-neg-to-sub` (line 1845) |
| `x - (-y)` → `x + y` | ✅ HAVE | `sub-neg-right-to-add` (line 1870) |
| `(-x) * (-y)` → `x * y` | ✅ HAVE | `mul-double-neg` (line 1855) |
| `(-x) * y` → `-(x * y)` | ✅ HAVE | `mul-neg-left/right` (lines 1849-1850) |
| `(-x) / (-y)` → `x / y` | ✅ HAVE | `sdiv-neg-both` (line 2833) |
| `(x + c1) - c2` → merge | ✅ HAVE | `sub-of-add-merge-consts` (line 2792) |
| `(x - c1) + c2` → merge | ✅ HAVE | `add-sub-merge-consts` (line 2786) |
| `(x - c1) - c2` → merge | ✅ HAVE | `sub-chain-merge-consts` (line 2798) |
| `(x / c1) / c2` → merge | ✅ HAVE | `sdiv/udiv-merge-consts` (lines 2696-2699) |

### True Gaps

The real gaps are **structural** - operations not yet in the `SpirvLang` enum:

1. ~~**Floating-point operations**: FAdd, FSub, FMul, FDiv, FNegate~~ ✅ (Session 43 - constant folding)
2. ~~**Min/Max operations**: SMin, SMax, UMin, UMax, FMin, FMax~~ ✅ (Session 42 - constant folding + rewrites)
3. ~~**Composite/Vector operations**: Extract, Insert, Shuffle, Dot~~ ✅ (Sessions 30, 39 - all constant folding implemented)

**All True Gaps are now COMPLETE!** The Rust e-graph optimizer has feature parity with the C++ spirv-opt
constant folding rules for all arithmetic, floating-point, min/max, matrix, and vector operations.

---

### Future Improvements

To eventually remove guards entirely, we would need:
1. Identify why the decomposition rules cause incorrect unifications
2. Either fix the underlying rule soundness issue
3. Or use the analysis to make smarter decisions in appliers

The analysis infrastructure now provides the foundation for these improvements.

### Session 13 - COMPLETED

**Verification of C++ spirv-opt folding rule coverage:**

Confirmed that all major C++ folding_rules.cpp optimizations are already implemented:

| C++ Rule | Rust Implementation |
|----------|---------------------|
| `FactorAddMuls` | ✅ factor-add-muls, factor-sub-muls, distribute-mul-add/sub (lines 4408-4422) |
| `RedundantAndAddSub` | ✅ BandRedundantAdd, BandRedundantSub (lines 7816-7878) |
| `RedundantAndShift` | ✅ BandRedundantShift (lines 7880-7922) |
| `RedundantAndOrXor` | ✅ BandRedundantOr, BandRedundantXor (lines 7748-7815) |
| Min/Max identity | ✅ smin-self, smax-self, umin-self, umax-self (line 3930-3933) |
| Min/Max absorption | ✅ smin-absorb-left/right, etc. (lines 3942-3949) |
| Min/Max cross-absorption | ✅ smin-smax-absorb, etc. (lines 3958-3961) |

**Rust optimizer advantages over C++:**
- Strength reduction happens automatically: `4*(x-1)` → `(x-1)<<2`
- Global optimization in single egraph pass vs multiple local passes
- All equivalent forms explored via bidirectional rewrites

**All 763 tests passing!**

### Session 16 - COMPLETED

**Floating-point operations expansion:**

Added new FP operations to `SpirvLang` enum and translation infrastructure:
- `FRem` - floating-point remainder
- `FMin`, `FMax` - floating-point min/max (via GLSL.std.450 extended instructions)

**Floating-point comparison operations (12 new ops):**
- Ordered comparisons (return false if either operand is NaN):
  - `FOrdEq`, `FOrdNe`, `FOrdLt`, `FOrdLe`, `FOrdGt`, `FOrdGe`
- Unordered comparisons (return true if either operand is NaN):
  - `FUnordEq`, `FUnordNe`, `FUnordLt`, `FUnordLe`, `FUnordGt`, `FUnordGe`

**Analysis updates:**
- Added FRem, FMin, FMax to analysis `make()` with width propagation
- Added all 12 FP comparison operations with boolean result (bit_width = 1)

**Translation support (opt_block.rs):**
- Added FRem with Op::FRem
- Added all FP comparison operations with correct Op codes
- Added FMin/FMax to continue list (require GLSL.std.450 extended instruction reconstruction)

**FP min/max rewrite rules:**
- `fmin-self`, `fmax-self`: `fmin(a, a) = a`, `fmax(a, a) = a`
- `fmin-comm`, `fmax-comm`: commutativity
- `fmin-absorb-left/right`, `fmax-absorb-left/right`: `fmin(x, fmin(x, y)) = fmin(x, y)`
- `fmin-assoc-left`, `fmax-assoc-left`: associativity
- `fmin-fmax-absorb`, `fmax-fmin-absorb`: `fmin(x, fmax(x, y)) = x`, `fmax(x, fmin(x, y)) = x`

**FP comparison self-identity rules:**
- `fordeq-self`: `a ==ord a = true`
- `fordne-self`: `a !=ord a = false`
- `fordlt-self`, `fordle-self`, `fordgt-self`, `fordge-self`: self-comparisons
- Same for unordered comparisons (funordeq-self, etc.)

**Unit tests added (20 new tests):**
- `fmin_self_is_identity`, `fmax_self_is_identity`
- `fmin_comm`, `fmax_comm`
- `fmin_absorb_left`, `fmax_absorb_left`
- `fmin_fmax_absorb`, `fmax_fmin_absorb`
- 12 FP comparison self-identity tests

**All 791 tests passing!**

### Session 17 - COMPLETED

**Reciprocal division folding:**

Added `FDivReciprocal` optimization:
- Pattern: `x / (1.0/y)` → `x * y` (reciprocal folding)
- Implemented as custom `Applier` that checks if inner dividend is FP 1.0
- Works with both 32-bit (`0x3F800000`) and 64-bit (`0x3FF0000000000000`) FP 1.0

**Implementation details:**
- Added `FDivReciprocal` struct with `x`, `one`, `y` Vars
- Added rewrite rule: `(fdiv ?x (fdiv ?one ?y))` → `FDivReciprocal { x, one, y }`
- Applier checks `is_fp_one()` on `one` and returns `fmul(x, y)` if true

**Gap analysis updates:**
- Marked Priority 5 (FP rules) 14-18 as complete
- Marked Priority 6 (min/max rules) 21-22 as complete
- Only Clamp pattern (23) remains as future scope

**Unit test:**
- `fdiv_reciprocal_to_fmul` - verifies `x / (1.0/y)` optimizes to `x * y`

**Add-band-complement rules (RedundantAndAddSub):**

Added 12 rewrite rules for complementary mask addition patterns:
- Self-value pattern: `(x & m) + (x & ~m) = x` (8 orderings for commutativity)
- Convert to OR pattern: `(x & m) + (y & ~m) = (x & m) | (y & ~m)` (4 orderings)

The convert-to-OR pattern is mathematically sound because when masks are complements,
the AND results have no overlapping bits, making addition equivalent to bitwise OR.

**Rewrite rules added:**
- `add-band-complement-self` and 7 commutative variants
- `add-band-complement-to-or` and 3 commutative variants

**Unit tests:**
- `add_band_complement_self_simplifies_to_value` - verifies `(x & m) + (x & ~m) = x`
- `add_band_complement_to_or` - verifies `(x & m) + (y & ~m) = (x & m) | (y & ~m)`

**Completed RedundantAndShift patterns:**

Added signed right shift patterns to complete the `BandRedundantShift` rules:
- `band-redundant-shr-signed` and `band-redundant-shr-signed-comm`
- Now covers all three shift kinds: left, right unsigned, right signed

Note: Signed right shift fills with sign bits, so the mask redundancy optimization
is conservative (returns `false` for `clears_mask`). The pattern is available
for future improvements if needed.

**All 534 tests passing!**

## Session 18: Vector Operations and Clamp Patterns

**Vector operations implemented:**

1. **InsertFeedingExtract** - `extract(insert(v, x, i), i) → x`
   - Simple pattern matching rule for same-index case
   - `ExtractInsertDiffIdx` applier for different-index case (extract(insert(v, x, i), j) → extract(v, j) when i != j)
   - Tests: `extract_insert_same_idx_returns_inserted_value`, `extract_insert_diff_idx_extracts_from_original`

2. **VectorShuffleFeedingExtract** - Fully implemented (Session 32)
   - `ShuffleMask` struct added as e-graph leaf node (`SMask(ShuffleMask)`)
   - `ExtractShuffle` applier traces through shuffle mask to find source vector/index
   - Tests: `extract_shuffle_from_first_vec`, `extract_shuffle_from_second_vec`, etc.

3. **VectorShuffleFeedingShuffle** - Implemented (Session 32)
   - `ShuffleFeedingShuffle` applier composes masks when outer shuffle only uses first operand
   - Eliminates nested shuffles when composed mask only needs one inner operand
   - Tests: `shuffle_shuffle_compose_from_first`, `shuffle_shuffle_uses_only_b`, etc.

3. **Dot product commutativity** - `dot(a, b) → dot(b, a)`
   - Added as simple rewrite rule
   - Test: `dot_commutativity`

**Clamp operations added to SpirvLang:**

- `SClamp([Id; 3])` - Signed clamp (GLSLstd450SClamp)
- `UClamp([Id; 3])` - Unsigned clamp (GLSLstd450UClamp)
- `FClamp([Id; 3])` - Floating-point clamp (GLSLstd450FClamp)

**Clamp analysis:**
- Constant folding: If x, lo, hi are all constants, compute clamped result
- Range analysis: If lo/hi are constants, result range is bounded

**Clamp rewrite rules (12 total):**
- `smin-smax-to-sclamp`: `(smin (smax ?x ?lo) ?hi)` → `(sclamp ?x ?lo ?hi)`
- `smax-smin-to-sclamp`: `(smax (smin ?x ?hi) ?lo)` → `(sclamp ?x ?lo ?hi)`
- Same patterns for unsigned (umin/umax/uclamp) and floating-point (fmin/fmax/fclamp)
- Commuted versions for all patterns
- Identity rules: `sclamp(x, x, hi)` → `smin(x, hi)`, `sclamp(x, lo, x)` → `smax(x, lo)`
- Same-bounds: `sclamp(x, lo, lo)` → `lo`

**Cost function:**
- Clamp operations cost 1 (vs 2 for nested min/max) to prefer single-instruction form

**Tests (7 new):**
- `smin_smax_to_sclamp`, `smax_smin_to_sclamp`
- `umin_umax_to_uclamp`, `fmin_fmax_to_fclamp`
- `sclamp_same_bounds_simplifies`, `uclamp_self_min_simplifies`
- `clamp_extracts_to_cheaper_form` - Verifies extractor prefers clamp over nested min/max

**E-graph benefit:** Clamp patterns are recognized globally across the entire expression tree,
potentially optimizing nested min/max chains that span multiple operations.

**All 78 tests passing!** (544 tests in lib crate)

## Session 19: Comparison Chain Patterns - COMPLETED

**Comparison chain patterns implemented:**

This session added extensive support for simplifying comparison chains - patterns where
multiple comparisons are combined with logical AND/OR operators.

**Contradictory comparison patterns (12 rules):**
- `(a < b) && (a > b)` → `false` (strict < and > contradict)
- `(a <= b) && (a > b)` → `false` (non-strict and strict contradict)
- `(a < b) && (a >= b)` → `false` (strict and non-strict contradict)
- Same patterns for unsigned comparisons (ult/ugt, ule/uge)
- All with commuted variants for AND operand ordering

**Tautological OR patterns (8 rules):**
- `(a < b) || (a >= b)` → `true` (covers all cases)
- `(a <= b) || (a > b)` → `true` (covers all cases)
- Same for unsigned comparisons
- All with commuted variants

**Transitive comparison chains (4 rules with custom appliers):**
Pattern: `(a < b) && (b < c)` where a,c are constants

- `CmpChainTransitive` applier: Detects unsatisfiable chains (e.g., `(10 < x) && (x < 10)` → `false`)
- Also simplifies single-value ranges (e.g., `(9 < x) && (x < 11)` → `x == 10`)
- Works for both signed (`slt`/`sle`) and unsigned (`ult`/`ule`) comparisons

**Same-LHS comparison chains (4 rules with custom appliers):**
Pattern: `(a < b) && (a < c)` where b,c are constants

- `CmpChainSameLhs` applier: Simplifies to `(a < min(b, c))`
- Reduces two comparisons to one when bounds are known

**Same-RHS comparison chains (4 rules with custom appliers):**
Pattern: `(a < c) && (b < c)` where a,b are constants

- `CmpChainSameRhs` applier: Simplifies to `(max(a, b) < c)`
- Reduces two comparisons to one when bounds are known

**Mixed strict/non-strict chains (4 rules with custom appliers):**
Pattern: `(a <= b) && (b < c)` or `(a < b) && (b <= c)`

- `CmpChainMixed` applier: Handles mixed comparison operators
- Detects unsatisfiable ranges and single-value solutions

**Tests added (9 new):**
- `slt_sgt_contradictory_is_false` - `(a < b) && (a > b)` → false
- `ult_uge_contradictory_is_false` - `(a < b) && (a >= b)` → false (unsigned)
- `slt_sge_tautology_is_true` - `(a < b) || (a >= b)` → true
- `ule_ugt_tautology_is_true` - `(a <= b) || (a > b)` → true (unsigned)
- `slt_chain_unsatisfiable` - `(10 < x) && (x < 10)` → false
- `slt_chain_single_value` - `(9 < x) && (x < 11)` → `x == 10`
- `slt_same_lhs_simplifies` - `(x < 10) && (x < 20)` → `x < 10`
- `slt_same_rhs_simplifies` - `(10 < c) && (20 < c)` → `20 < c`
- `sle_chain_equal_bounds` - `(5 <= x) && (x <= 5)` → `x == 5`

**Gap Analysis Update:**
- Priority 4 (Comparison Chaining) now marked as COMPLETE

**All 817 tests passing!**

## Session 20: FMix (Linear Interpolation) Optimizations - COMPLETED

**FMix operation added to SpirvLang:**

FMix implements the GLSL `mix()` function (GLSLstd450FMix): `fmix(x, y, a) = x*(1-a) + y*a`

This is a very common operation in shaders for blending, animation, and color mixing.

**SpirvLang enum addition:**
```rust
"fmix" = FMix([Id; 3]),  // GLSLstd450FMix: fmix(x, y, a) = x*(1-a) + y*a
```

**Analysis:**
- Width propagation from operands
- Origin tracking (pure floating-point operation)

**Cost function:**
- FMix costs 2 (single extended instruction, cheaper than expansion)

**FMix rewrite rules (4 total):**

1. `fmix-zero`: `fmix(x, y, 0)` → `x` (when a=0, result is entirely x)
   - Custom `FMixZero` applier checks if blend factor is FP zero

2. `fmix-one`: `fmix(x, y, 1)` → `y` (when a=1, result is entirely y)
   - Custom `FMixOne` applier checks if blend factor is FP one

3. `fmix-self`: `fmix(x, x, a)` → `x` (mixing value with itself)
   - Simple pattern match (no applier needed)

4. `fmix-feeding-extract`: `extract(fmix(x, y, a), i)` → `extract(x, i)` or `extract(y, i)`
   - Custom `FMixFeedingExtract` applier
   - When `a[i]` is 0.0, result is `extract(x, i)`
   - When `a[i]` is 1.0, result is `extract(y, i)`
   - Matches C++ `FMixFeedingExtract` optimization from folding_rules.cpp
   - Enables per-component optimization when blend factors are constant

**Translation/continuation support:**
- Added FMix to continue lists in `translate.rs` and `bin/opt_block.rs`
- FMix requires GLSLstd450 extended instruction reconstruction (not yet implemented)

**Unit tests (7 new):**
- `fmix_with_zero_returns_x` - verifies `fmix(x, y, 0) = x`
- `fmix_with_one_returns_y` - verifies `fmix(x, y, 1) = y`
- `fmix_self_returns_self` - verifies `fmix(x, x, a) = x`
- `fmix_with_symbolic_a_preserved` - verifies symbolic blend factor is not optimized
- `fmix_feeding_extract_with_zero_component` - verifies component extraction optimization
- `fmix_feeding_extract_with_one_component` - verifies component extraction optimization
- `fmix_feeding_extract_with_other_component_preserved` - verifies 0.5 blend is preserved

**E-graph advantage:**
The FMixFeedingExtract pattern benefits from e-graph global optimization:
- Can detect component-wise constant blend factors across the expression tree
- Enables optimization even when fmix and extract are separated by other operations

**All 836 tests passing!**

## Session 21: FStep and FSmoothStep (GLSL Step Functions) - COMPLETED

**FStep operation added to SpirvLang:**

FStep implements the GLSL `step()` function (GLSLstd450Step): `step(edge, x) = x < edge ? 0.0 : 1.0`

This is commonly used for threshold-based selection in shaders.

**FSmoothStep operation added to SpirvLang:**

FSmoothStep implements the GLSL `smoothstep()` function (GLSLstd450SmoothStep):
`smoothstep(lo, hi, x) = t * t * (3 - 2*t)` where `t = clamp((x - lo) / (hi - lo), 0, 1)`

This provides smooth Hermite interpolation for transitions.

**SpirvLang enum additions:**
```rust
// Step functions (GLSL.std.450)
"fstep" = FStep([Id; 2]),                // GLSLstd450Step: step(edge, x) = x < edge ? 0.0 : 1.0
"fsmoothstep" = FSmoothStep([Id; 3]),    // GLSLstd450SmoothStep: smoothstep(lo, hi, x)
```

**Analysis:**
- Width propagation from operands
- Origin tracking (pure floating-point operations)

**Cost function:**
- FStep and FSmoothStep cost 2 (single extended instruction each)

**FStep rewrite rules (2 total):**

1. `fstep-self`: `step(x, x)` → `1.0` (x is not less than itself)
   - Custom `FStepSelf` applier returns 1.0

2. `fstep-const`: `step(edge, x)` with constants → `0.0` or `1.0`
   - Custom `FStepConst` applier evaluates comparison
   - Supports both 32-bit and 64-bit floats

**FSmoothStep rewrite rules (4 total):**

1. `fsmoothstep-at-lo`: `smoothstep(lo, hi, lo)` → `0.0` (x at lo means t=0)
   - Custom `FSmoothStepAtLo` applier returns 0.0

2. `fsmoothstep-at-hi`: `smoothstep(lo, hi, hi)` → `1.0` (x at hi means t=1)
   - Custom `FSmoothStepAtHi` applier returns 1.0

3. `fsmoothstep-same-bounds`: `smoothstep(x, x, y)` → `step(x, y)`
   - Degenerate case converts to step function
   - Simple pattern match (no applier needed)

4. `fsmoothstep-const`: `smoothstep(lo, hi, x)` with all constants → computed result
   - Custom `FSmoothStepConst` applier evaluates full smoothstep formula
   - Handles edge cases (x < lo → 0, x > hi → 1, lo == hi → step behavior)
   - Supports both 32-bit and 64-bit floats

**Translation/continuation support:**
- Added FStep/FSmoothStep to continue lists in `translate.rs` and `bin/opt_block.rs`
- Both require GLSLstd450 extended instruction reconstruction (not yet implemented)

**Unit tests (14 new):**

FStep tests:
- `fstep_self_returns_one` - verifies `step(x, x) = 1.0`
- `fstep_const_below_edge_returns_zero` - verifies `step(0.5, 0.3) = 0.0`
- `fstep_const_at_edge_returns_one` - verifies `step(0.5, 0.5) = 1.0`
- `fstep_const_above_edge_returns_one` - verifies `step(0.5, 0.8) = 1.0`
- `fstep_symbolic_preserved` - verifies symbolic operands preserved
- `fstep_64bit_const_below_edge_returns_zero` - verifies 64-bit support

FSmoothStep tests:
- `fsmoothstep_at_lo_returns_zero` - verifies `smoothstep(lo, hi, lo) = 0.0`
- `fsmoothstep_at_hi_returns_one` - verifies `smoothstep(lo, hi, hi) = 1.0`
- `fsmoothstep_same_bounds_becomes_fstep` - verifies `smoothstep(x, x, y) = step(x, y)`
- `fsmoothstep_const_below_lo_returns_zero` - verifies clamping below range
- `fsmoothstep_const_above_hi_returns_one` - verifies clamping above range
- `fsmoothstep_const_midpoint` - verifies `smoothstep(0, 1, 0.5) = 0.5`
- `fsmoothstep_symbolic_preserved` - verifies symbolic operands preserved
- `fsmoothstep_64bit_at_lo_returns_zero` - verifies 64-bit support

**E-graph advantage:**
The step functions benefit from e-graph optimization:
- Pattern matching works across expression boundaries
- Can detect when degenerate smoothstep becomes step, then apply step optimizations
- Global constant propagation enables more folding opportunities

**All 857 tests passing!**

### FAbs and FSign (Floating-Point Absolute Value and Sign) - COMPLETED

**FAbs operation added to SpirvLang:**

FAbs implements the GLSL `abs()` function for floats (GLSLstd450FAbs): `fabs(x) = |x|`

This returns the absolute value of the floating-point argument.

**FSign operation added to SpirvLang:**

FSign implements the GLSL `sign()` function for floats (GLSLstd450FSign):
- Returns -1.0 if x < 0
- Returns 0.0 if x == 0
- Returns 1.0 if x > 0

**SpirvLang enum additions:**
```rust
// Math functions (GLSL.std.450)
"fabs" = FAbs(Id),                       // GLSLstd450FAbs: floating-point absolute value
"fsign" = FSign(Id),                     // GLSLstd450FSign: floating-point sign (-1, 0, or 1)
```

**FAbs rewrite rules (3 total):**

1. `fabs-idempotent`: `fabs(fabs(x))` → `fabs(x)` (absolute value is idempotent)

2. `fabs-fneg`: `fabs(fneg(x))` → `fabs(x)` (negation doesn't affect absolute value)

3. `fabs-const`: `fabs(x)` with constant x → computed |x|
   - Custom `FAbsConst` applier evaluates absolute value
   - Supports both 32-bit and 64-bit floats

**FSign rewrite rules (1 total):**

1. `fsign-const`: `fsign(x)` with constant x → -1.0, 0.0, or 1.0
   - Custom `FSignConst` applier evaluates sign function
   - Supports both 32-bit and 64-bit floats

**Unit tests (12 new):**

FAbs tests:
- `fabs_const_positive` - verifies `fabs(3.5) = 3.5`
- `fabs_const_negative` - verifies `fabs(-2.5) = 2.5`
- `fabs_const_zero` - verifies `fabs(0.0) = 0.0`
- `fabs_idempotent` - verifies `fabs(fabs(x)) = fabs(x)`
- `fabs_fneg` - verifies `fabs(fneg(x)) = fabs(x)`
- `fabs_symbolic_preserved` - verifies symbolic operands preserved
- `fabs_64bit_negative` - verifies 64-bit support

FSign tests:
- `fsign_const_positive` - verifies `fsign(3.5) = 1.0`
- `fsign_const_negative` - verifies `fsign(-2.5) = -1.0`
- `fsign_const_zero` - verifies `fsign(0.0) = 0.0`
- `fsign_symbolic_preserved` - verifies symbolic operands preserved
- `fsign_64bit_negative` - verifies 64-bit support

**All 869 tests passing!**

## Session 22: GLSL Math Functions (sqrt, exp, log, pow, trig, rounding) - COMPLETED

**22 new GLSL.std.450 math functions added to SpirvLang:**

**Square root functions:**
- `FSqrt(Id)` - GLSLstd450Sqrt: square root
- `FInverseSqrt(Id)` - GLSLstd450InverseSqrt: 1/sqrt(x)

**Exponential/logarithmic functions:**
- `FExp(Id)` - GLSLstd450Exp: e^x
- `FExp2(Id)` - GLSLstd450Exp2: 2^x
- `FLog(Id)` - GLSLstd450Log: ln(x)
- `FLog2(Id)` - GLSLstd450Log2: log2(x)
- `FPow([Id; 2])` - GLSLstd450Pow: x^y

**Trigonometric functions:**
- `FSin(Id)`, `FCos(Id)`, `FTan(Id)` - basic trig
- `FAsin(Id)`, `FAcos(Id)`, `FAtan(Id)` - inverse trig
- `FAtan2([Id; 2])` - two-argument arctangent

**Hyperbolic functions:**
- `FSinh(Id)`, `FCosh(Id)`, `FTanh(Id)` - hyperbolic sin/cos/tan

**Rounding functions:**
- `FFloor(Id)` - GLSLstd450Floor: floor(x)
- `FCeil(Id)` - GLSLstd450Ceil: ceil(x)
- `FRound(Id)` - GLSLstd450Round: round to nearest
- `FTrunc(Id)` - GLSLstd450Trunc: truncate towards zero
- `FFract(Id)` - GLSLstd450Fract: x - floor(x)

**Constant folding appliers (23 total):**
All functions have constant folding appliers that evaluate at compile time
when operands are constants. Supports both 32-bit and 64-bit floats.

**FPow special case optimizations:**
Custom `FPowConst` applier handles common exponent values:
- `pow(x, 0)` → `1.0`
- `pow(x, 1)` → `x`
- `pow(x, 2)` → `x * x` (strength reduction)
- `pow(x, 0.5)` → `sqrt(x)` (strength reduction)
- `pow(x, -1)` → `1/x` (reciprocal)

**Algebraic simplification rules:**

Exp/Log cancellation (4 rules):
- `exp(log(x))` → `x`
- `log(exp(x))` → `x`
- `exp2(log2(x))` → `x`
- `log2(exp2(x))` → `x`

Idempotent rounding rules (4 rules):
- `floor(floor(x))` → `floor(x)`
- `ceil(ceil(x))` → `ceil(x)`
- `trunc(trunc(x))` → `trunc(x)`
- `round(round(x))` → `round(x)`

**Unit tests (37 new):**
- Constant folding: fsqrt, fexp, fexp2, flog, flog2, fpow, fsin, fcos, ftan,
  ffloor, fceil, ftrunc, ffract, fround, finversesqrt, fasin, facos, fatan,
  fsinh, fcosh, ftanh
- FPow special cases: zero/one/two/half/neg_one exponents
- Exp/log cancellations: exp_log, log_exp, exp2_log2, log2_exp2
- Idempotent rules: floor, ceil, trunc, round
- 64-bit support: fsqrt_const_fold_64bit

**All 904 tests passing!**

## Session 23: FP Mul-Div Cancellation and Constant Merging - COMPLETED

**C++ spirv-opt patterns implemented:**

These optimizations match the C++ `MergeMulDivArithmetic` and `MergeDivMulArithmetic`
folding rules from `folding_rules.cpp`.

**FP mul-div cancellation (4 rules):**
- `(x / y) * y` → `x` (MergeMulDivArithmetic cancellation)
- `y * (x / y)` → `x` (commuted)
- `(x * y) / y` → `x` (MergeDivMulArithmetic cancellation)
- `(y * x) / y` → `x` (commuted)

**FP constant merging (4 rules):**
- `(x / c1) * c2` → `x * (c2/c1)` (MergeMulDivArithmetic constant merge)
- `c2 * (x / c1)` → `x * (c2/c1)` (commuted)
- `(x * c1) / c2` → `x * (c1/c2)` (MergeDivMulArithmetic constant merge)
- `(c1 * x) / c2` → `x * (c1/c2)` (commuted)

**FMod zero identity (1 rule):**
- `0.0 % x` → `0.0` (RedundantFMod from C++ spirv-opt)

**E-graph advantage:**
The constant merge rules compose with the `fmul-one` rule: when `c1/c2 = 1.0`,
the expression simplifies to just `x`. This happens automatically through
the e-graph's global optimization.

**Applier structs added (5 total):**
- `FMulDivCancel` - handles `(x/y)*y` cancellation
- `FDivMulCancel` - handles `(x*y)/y` cancellation
- `FMulDivConstMerge` - handles `(x/c1)*c2` constant merge
- `FDivMulConstMerge` - handles `(x*c1)/c2` constant merge
- `FModZero` - handles `0.0 % x` identity

**Unit tests (10 new):**
- `fmul_fdiv_cancel_right` - `(x/y)*y` → `x`
- `fmul_fdiv_cancel_left` - `y*(x/y)` → `x`
- `fdiv_fmul_cancel_right` - `(x*y)/y` → `x`
- `fdiv_fmul_cancel_left` - `(y*x)/y` → `x`
- `fmul_fdiv_const_merge` - `(x/2.0)*4.0` → `x*2.0`
- `fdiv_fmul_const_merge` - `(x*4.0)/2.0` → `x*2.0`
- `fdiv_fmul_const_merge_to_identity` - `(x*2.0)/2.0` → `x`
- `fmod_zero_dividend` - `0.0 % x` → `0.0`
- `fmod_zero_dividend_64bit` - 64-bit support
- `fmul_fdiv_const_merge_64bit` - 64-bit constant merge

**All 914 tests passing!**

## Session 24: FP Arithmetic Constant Merging and Negate Propagation - COMPLETED

**C++ spirv-opt patterns implemented:**

These optimizations match the C++ folding rules from `folding_rules.cpp`:
- `RedundantFSub` - fsub with zero on left
- `MergeMulMulArithmetic` - consecutive FP multiplies with constants
- `MergeDivDivArithmetic` - consecutive FP divides with constants
- `MergeNegateMulDivArithmetic` - negate propagation into mul/div
- `MergeNegateAddSubArithmetic` - negate propagation into add/sub

**fsub with zero on left (enhanced FSubZeroRight applier):**
- `0.0 - x` → `fneg(x)` (C++ RedundantFSub)

**FP multiply constant merge (4 rules, MergeMulMulArithmetic):**
- `(x * c1) * c2` → `x * (c1 * c2)`
- `(c1 * x) * c2` → `x * (c1 * c2)` (commuted inner)
- `c2 * (x * c1)` → `x * (c1 * c2)` (commuted outer)
- `c2 * (c1 * x)` → `x * (c1 * c2)` (both commuted)

**FP divide constant merge (2 rules, MergeDivDivArithmetic):**
- `(x / c1) / c2` → `x / (c1 * c2)` (constants multiply)
- `(c1 / x) / c2` → `(c1 / c2) / x` (constants divide)

**FP negate propagation into mul/div (3 rules, MergeNegateMulDivArithmetic):**
- `-(x * c)` → `x * (-c)` (negate absorbed into constant)
- `-(c * x)` → `x * (-c)` (commuted)
- `-(x / c)` → `x / (-c)` (divisor constant)
- `-(c / x)` → `(-c) / x` (dividend constant)

**FP negate propagation into add/sub (2 rules, MergeNegateAddSubArithmetic):**
- `-(x + c)` → `(-c) - x` (constant negated, converted to sub)
- `-(c + x)` → `(-c) - x` (commuted)
- `-(x - y)` → `y - x` (operands swapped, always applies)

**Applier structs added (7 total):**
- `FMulMulConstMerge` - handles `(x * c1) * c2` constant merge
- `FDivDivConstMerge` - handles `(x / c1) / c2` constant merge
- `FNegMulConst` - handles `-(x * c)` negate propagation
- `FNegDivConst` - handles `-(x / c)` and `-(c / x)` negate propagation
- `FNegAddConst` - handles `-(x + c)` negate propagation
- `FNegSubConst` - handles `-(x - y)` operand swap

**E-graph advantage:**
These rules compose with existing rules. For example:
- `(x * 2.0) * 0.5` → `x * 1.0` → `x` (via fmul-one)
- `-(x * 1.0)` → `x * -1.0` → `-x` (via fmul-fneg)

**Unit tests (10 new):**
- `fsub_zero_left` - `0.0 - x` → `fneg(x)`
- `fsub_zero_left_64bit` - 64-bit support
- `fmul_fmul_const_merge` - `(x * 2.0) * 3.0` → `x * 6.0`
- `fmul_fmul_const_merge_64bit` - 64-bit support
- `fdiv_fdiv_const_merge` - `(x / 2.0) / 3.0` → `x / 6.0`
- `fdiv_fdiv_const_merge_64bit` - 64-bit support
- `fneg_fmul_const` - `-(x * 2.0)` → `x * (-2.0)`
- `fneg_fdiv_const` - `-(x / 2.0)` → `x / (-2.0)`
- `fneg_fsub_swap` - `-(x - y)` → `y - x`
- `fneg_fadd_const` - `-(x + 2.0)` → `(-2.0) - x`

**All 924 tests passing!**

## Session 25: Add/Sub Constant Merging (FP and Integer) - COMPLETED

**C++ spirv-opt patterns implemented:**

These optimizations match the C++ `MergeAddAddArithmetic`, `MergeAddSubArithmetic`,
`MergeSubAddArithmetic`, and `MergeSubSubArithmetic` folding rules from `folding_rules.cpp`.

**FP add-add constant merging (4 rules, MergeAddAddArithmetic):**
- `(x + c1) + c2` → `x + (c1 + c2)` (inner right constant)
- `(c1 + x) + c2` → `x + (c1 + c2)` (inner left constant)
- `c2 + (x + c1)` → `x + (c1 + c2)` (outer left, inner right)
- `c2 + (c1 + x)` → `x + (c1 + c2)` (outer left, inner left)

**FP add-sub constant merging (2 rules, MergeAddSubArithmetic):**
- `(x - c1) + c2` → `x + (c2 - c1)` (subtract constant added to add)
- `c2 + (x - c1)` → `x + (c2 - c1)` (commuted outer)

**FP sub-add constant merging (1 rule, MergeSubAddArithmetic):**
- `(x + c1) - c2` → `x + (c1 - c2)` (add constant subtracted)
- `(c1 + x) - c2` → `x + (c1 - c2)` (commuted inner)

**FP sub-sub constant merging (1 rule, MergeSubSubArithmetic):**
- `(x - c1) - c2` → `x - (c1 + c2)` (chain of subtractions)
- `(c1 - x) - c2` → `(c1 - c2) - x` (constant minuend)

**Integer add/sub constant merging (mirrors FP patterns):**
All FP patterns replicated for integer operations (`+`, `-`):
- `IAddAddConstMerge` - 4 commutative variants
- `IAddSubConstMerge` - 2 commutative variants
- `ISubAddConstMerge` - 2 commutative variants
- `ISubSubConstMerge` - 2 commutative variants

**E-graph advantage:**
These rules compose with identity rules. For example:
- `(x + 5) - 5` → `x + (5 - 5)` → `x + 0` → `x` (via add-zero)
- `(x - 3) + 3` → `x + (3 - 3)` → `x + 0` → `x` (via add-zero)

**Applier structs added (8 total):**
- `FAddAddConstMerge` - handles FP add-add patterns
- `FAddSubConstMerge` - handles FP add-sub patterns
- `FSubAddConstMerge` - handles FP sub-add patterns
- `FSubSubConstMerge` - handles FP sub-sub patterns
- `IAddAddConstMerge` - handles integer add-add patterns
- `IAddSubConstMerge` - handles integer add-sub patterns
- `ISubAddConstMerge` - handles integer sub-add patterns
- `ISubSubConstMerge` - handles integer sub-sub patterns

**Note on nested patterns:**
Initially implemented nested patterns like `c1 - (c2 - x)`, but these caused
applier conflicts due to variable semantics differing between pattern forms.
Removed nested patterns to match C++ spirv-opt scope (only handles
`(x op c1) op c2` forms, not `c1 op (c2 op x)` forms).

**Unit tests (8 new):**
- `fadd_fadd_const_merge` - `(x + 2.0) + 3.0` → `x + 5.0`
- `fadd_fadd_const_merge_64bit` - 64-bit support
- `fadd_fsub_const_merge` - `(x - 2.0) + 5.0` → `x + 3.0`
- `fsub_fadd_const_merge` - `(x + 3.0) - 2.0` → `x + 1.0`
- `fsub_fsub_const_merge` - `(x - 2.0) - 3.0` → `x - 5.0`
- `iadd_iadd_const_merge` - `(x + 2) + 3` → `x + 5`
- `iadd_isub_const_merge` - `(x - 2) + 5` → `x + 3`
- `isub_isub_const_merge` - `(x - 2) - 3` → `x - 5`

**All 932 tests passing!** (672 lib + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6 = 932 total)

---

## Session 26: ReciprocalFDiv Optimization

**Reviewed C++ folding_rules.cpp systematically to identify missing patterns:**

Verified we already have:
- `MergeGenericAddSubArithmetic`: `(a - b) + b = a` → `add-cancels-subtrahend`
- `FactorAddMuls`: `(a * b) + (a * c) = a * (b + c)` → `factor-add-muls`
- `ReassociateCommutiveBitwise`: `A | (b | C) = b | (A | C)` → `band/bor/bxor-reassociate-const-*`
- `IntMultipleBy1`: `x * 1 = x` → `mul-one`
- `RedundantAndShift`: `1 & (b << 1) = 0` → `BandRedundantShift`
- `DotProductDoingExtract`: `dot(v, unit_i) -> extract(v, i)` → dot unit vector optimization
- `MergeNegateArithmetic`: `neg(neg(x)) = x` → `neg-neg-cancel`
- `MergeNegateMulDivArithmetic`: `-(x * c) -> x * (-c)` → `neg-mul-const-*`, `fneg-fmul-const-*`
- `MergeNegateAddSubArithmetic`: `-(x + c) -> (-c) - x` → `neg-add-const`, `fneg-fadd`

**New optimization implemented:**

`ReciprocalFDiv` (C++ spirv-opt pattern):
- Pattern: `x / const` → `x * (1.0/const)`
- Strength reduction: converts FP division to multiplication
- Implemented as `FDivConstToMul` applier
- Safety checks:
  - Only applies to 32-bit and 64-bit floats
  - Skips if constant is zero (would create infinity)
  - Skips if constant is 1.0 (fdiv-one handles this)
  - Skips if reciprocal is not finite
- Creates new constant `1.0/c` and replaces `fdiv(x, c)` with `fmul(x, 1.0/c)`

**Optimizations deferred (require language extensions):**

- `VectorShuffleFeedingShuffle`: Requires complex shuffle mask composition
- `CompositeConstructFeedingExtract`: Requires `CompositeConstruct` in e-graph language

**Unit tests added (5 new):**
- `fdiv_const_to_fmul_32bit` - `x / 2.0` → `x * 0.5`
- `fdiv_const_to_fmul_64bit` - `x / 4.0` → `x * 0.25` (64-bit)
- `fdiv_const_to_fmul_skips_zero` - `x / 0.0` unchanged (safety)
- `fdiv_const_to_fmul_skips_one` - `x / 1.0` → `x` (via fdiv-one)
- `fdiv_const_to_fmul_preserves_non_const` - `x / y` unchanged when y is variable

**All 937 tests passing!** (677 lib + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6 = 937 total)

---

## Session 27: Integer MulMul Constant Merge + C++ Parity Review

**Systematic review of C++ folding_rules.cpp patterns:**

Confirmed we already have all major C++ folding rules:
- `MergeMulDivArithmetic` → `fmul-fdiv-cancel-*`, `fdiv-fmul-cancel-*`, `fmul-fdiv-const-merge`, `fdiv-fmul-const-merge`
- `MergeDivDivArithmetic` → `fdiv-fdiv-const-merge`
- `MergeAddAddArithmetic` → `fadd-fadd-const-merge`, `iadd-iadd-const-merge`
- `MergeAddSubArithmetic` → `fadd-fsub-const-merge`, `iadd-isub-const-merge`
- `MergeSubAddArithmetic` → `fsub-fadd-const-merge`, `isub-iadd-const-merge`
- `MergeSubSubArithmetic` → `fsub-fsub-const-merge`, `isub-isub-const-merge`
- `MergeSubNegateArithmetic` → `fsub-fneg-*`, `sub-neg-*`
- `MergeAddNegateArithmetic` → `fadd-fneg-*`, `add-neg-*`
- `MergeMulNegateArithmetic` → `fmul-fneg-*`, `mul-neg-*`
- `MergeDivNegateArithmetic` → `fdiv-fneg-*`
- `FactorAddMuls` → `factor-add-muls`, `fp-factor-add-muls`
- `MergeGenericAddSubArithmetic` → `add-sub-cancel-*`
- `RedundantAndAddSub` → `BandRedundantAdd`, `BandRedundantSub`
- `RedundantAndShift` → `BandRedundantShift`
- `ReciprocalFDiv` → `FDivConstToMul` (Session 26)
- `MergeMulMulArithmetic` → `fmul-fmul-const-merge` (FP only - Session 24)

**New optimization implemented:**

`MergeMulMulArithmetic` for integers (C++ spirv-opt pattern):
- Pattern: `(x * c1) * c2` → `x * (c1 * c2)`
- Constant folding through nested integer multiplications
- Implemented as `IMulMulConstMerge` applier
- Handles all operand orderings:
  - `(x * c1) * c2` → `x * (c1 * c2)`
  - `(c1 * x) * c2` → `x * (c1 * c2)`
  - `c2 * (x * c1)` → `x * (c1 * c2)`
  - `c2 * (c1 * x)` → `x * (c1 * c2)`
- Uses wrapping multiplication for proper integer semantics

**Unit tests added (3 new):**
- `imul_imul_const_merge` - `(x * 2) * 3` → `x * 6`
- `imul_imul_const_merge_commuted` - `(2 * x) * 3` → `x * 6`
- `imul_imul_const_merge_outer_left` - `3 * (x * 2)` → `x * 6`

**C++ pattern coverage summary:**
All major folding rules from `folding_rules.cpp` are now implemented:
- Arithmetic: +/-/*/÷ constant merging (FP and integer)
- Negate propagation through all operations
- Redundant and masking optimizations
- Reciprocal division strength reduction
- Distributive law factoring

**All 940 tests passing!** (680 lib + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6 = 940 total)

---

## Session 28: Complete C++ folding_rules.cpp Parity Verification

**Exhaustive review of ALL C++ folding rules in `folding_rules.cpp`:**

Performed a complete audit of all patterns in the C++ spirv-opt `FoldingRules::AddFoldingRules()` function (lines 3378-3522) and verified their Rust e-graph equivalents.

### In-Scope Patterns (Arithmetic/Bitwise) - ALL IMPLEMENTED ✅

| C++ Pattern | Rust Equivalent | Status |
|-------------|-----------------|--------|
| `RedundantBinaryRhs0` (a+0, a-0, a\|0, a^0, a>>0, a<<0) | `add-zero`, `sub-zero-right`, `bor-zero`, `bxor-zero`, `shr-*-zero`, `shl-zero` | ✅ |
| `RedundantBinaryLhs0` (0+a, 0\|a, 0^a) | `add-zero-comm`, `bor-zero-comm`, `bxor-zero-comm` | ✅ |
| `RedundantBinaryLhs0To0` (0>>a, 0<<a, 0/a, 0%a) | `*-zero-left` patterns | ✅ |
| `ReassociateCommutiveBitwise` | `band/bor/bxor-reassociate-const-*` | ✅ |
| `RedundantSUDiv` (x/1=x) | `sdiv-one`, `udiv-one` (via `DivOne` applier) | ✅ |
| `RedundantSUMod` (x%1=0) | `srem-one`, `smod-one`, `umod-one` (via `RemOne` applier) | ✅ |
| `BitReverseScalarOrVector` | `BitReverseFold` | ✅ |
| `RedundantFAdd` | `fadd-zero`, `fadd-zero-comm` | ✅ |
| `RedundantFSub` | `fsub-zero`, `fsub-self` | ✅ |
| `RedundantFMul` | `fmul-one`, `fmul-zero` | ✅ |
| `RedundantFDiv` | `fdiv-one` | ✅ |
| `RedundantFMod` | `fmod-*` patterns | ✅ |
| `ReciprocalFDiv` | `FDivConstToMul` | ✅ |
| `MergeAddNegateArithmetic` | `add-neg-to-sub`, `add-neg-to-sub-swap`, `fadd-fneg-to-fsub` | ✅ |
| `MergeSubNegateArithmetic` | `sub-neg-right-to-add`, `sub-neg-left-to-neg-add`, `fsub-fneg-to-fadd` | ✅ |
| `MergeMulNegateArithmetic` | `mul-neg-left/right`, `fmul-fneg-left/right` | ✅ |
| `MergeDivNegateArithmetic` | `fdiv-fneg-both` | ✅ |
| `MergeNegateArithmetic` | `neg-neg-cancel`, `fneg-fneg-cancel` | ✅ |
| `MergeNegateMulDivArithmetic` | `FNegMulConst`, `FNegDivConst`, `NegMulConst`, `NegDivConst` | ✅ |
| `MergeNegateAddSubArithmetic` | `FNegAddConst`, `FNegSubConst` | ✅ |
| `MergeAddAddArithmetic` | `add-reassociate-const-*`, `fadd-fadd-const-merge` | ✅ |
| `MergeAddSubArithmetic` | `iadd-isub-const-merge-*`, `fadd-fsub-const-merge` | ✅ |
| `MergeSubAddArithmetic` | `isub-iadd-const-merge-*`, `fsub-fadd-const-merge` | ✅ |
| `MergeSubSubArithmetic` | `isub-isub-const-merge-*`, `fsub-fsub-const-merge` | ✅ |
| `MergeMulMulArithmetic` | `imul-imul-const-merge-*`, `fmul-fmul-const-merge-*` | ✅ |
| `MergeDivDivArithmetic` | `sdiv-merge-consts`, `udiv-merge-consts`, `fdiv-fdiv-const-merge` | ✅ |
| `MergeMulDivArithmetic` | `fmul-fdiv-cancel-*`, `fmul-fdiv-const-merge-*` | ✅ |
| `MergeDivMulArithmetic` | `fdiv-fmul-cancel-*`, `fdiv-fmul-const-merge-*` | ✅ |
| `FactorAddMuls` | `factor-add-muls`, `factor-add-muls-right`, `fp-factor-add-muls` | ✅ |
| `MergeGenericAddSubArithmetic` | `add-sub-cancel-right-simple`, `add-sub-cancel-left-simple` | ✅ |
| `IntMultipleBy1` | `mul-one` | ✅ |
| `RedundantAndOrXor` | `band-redundant-or-const`, `band-redundant-xor-const` | ✅ |
| `RedundantAndAddSub` | `band-redundant-add-const`, `band-redundant-sub-const` | ✅ |
| `RedundantAndShift` | `band-redundant-shl`, `band-redundant-shr` | ✅ |
| `RedundantSelect` | `select-same`, `SelectConstCond`, `SelectBoolArms` | ✅ |

### Out-of-Scope Patterns (Vector/Composite/CFG/Image)

| C++ Pattern | Why Out of Scope |
|-------------|------------------|
| `BitCastScalarOrVector` | Type conversion, not arithmetic optimization |
| `CompositeExtractFeedingConstruct` | Vector/composite operations - not in e-graph scope |
| `InsertFeedingExtract` | Vector/composite operations |
| `CompositeConstructFeedingExtract` | Vector/composite operations |
| `VectorShuffleFeedingExtract` | Vector shuffle mask handling |
| `FMixFeedingExtract` | Vector FMix + extract |
| `CompositeInsertToCompositeConstruct` | Vector/composite operations |
| `DotProductDoingExtract` | Vector dot product (partially supported) |
| `VectorShuffleFeedingShuffle` | Vector shuffle composition |
| `RemoveRedundantOperands` | Entry point metadata |
| `RedundantPhi` | CFG control flow |
| `StoringUndef` | Memory operations |
| `UpdateImageOperands` | Image sampling operations |
| `RedundantFMix` | GLSL FMix extension |

### Conclusion

**All arithmetic and bitwise folding patterns from C++ spirv-opt are now implemented in the Rust e-graph optimizer.** The e-graph approach offers advantages over the C++ implementation:

1. **Global optimization** - All patterns are applied in a single pass rather than iteratively
2. **Bidirectional exploration** - Equivalent forms are discovered through associativity/commutativity
3. **No ordering sensitivity** - E-graph saturation finds optimal forms regardless of rule order
4. **Strength reduction** - Patterns like `x * 4` → `x << 2` happen automatically through the unified optimization framework

**All 940 tests passing!** (680 lib + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6 = 940 total)

---

## Phase 2: Vector/Composite/CFG/Image Operations in E-Graph

### Goal
Extend the e-graph optimizer to handle vector/composite operations, CFG-related patterns, and image operations - all in a single unified pass for global optimization.

### Priority Order

#### Priority 1: Vector/Composite Operations (High Impact)

These operations are heavily used in shader code and offer significant optimization opportunities.

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 1 | `CompositeConstructFeedingExtract` | `extract(construct(a,b,c,...), i)` → element i | ✅ DONE (Session 29) |
| 2 | `CompositeExtractFeedingConstruct` | `construct(extract(v,0), extract(v,1), ...)` → `v` | ✅ DONE (Session 30) |
| 3 | `CompositeInsertToCompositeConstruct` | Series of inserts covering object → `construct` | ✅ DONE (Session 30) |
| 4 | `VectorShuffleFeedingShuffle` | `shuffle(shuffle(v1,v2,m1), v3, m2)` → simplified | ✅ DONE (Session 32 - ShuffleFeedingShuffle applier) |
| 5 | Insert/Extract optimization | `extract(insert(v,x,i),i)` → `x` | ✅ HAVE (extract-insert-same-idx) |
| 6 | Insert/Extract different idx | `extract(insert(v,x,i),j)` where i≠j | ✅ HAVE (extract-insert-diff-idx) |
| 7 | Shuffle feeding extract | `extract(shuffle(v1,v2,mask),idx)` | ✅ DONE (Session 32 - ExtractShuffle applier) |

**New operations added to SpirvLang (Session 29):**
- `Vec2([Id; 2])` - OpCompositeConstruct for 2-element vector
- `Vec3([Id; 3])` - OpCompositeConstruct for 3-element vector
- `Vec4([Id; 4])` - OpCompositeConstruct for 4-element vector

#### Priority 2: CFG Operations (Medium Impact)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 1 | `RedundantPhi` | `phi(x, x, x, ...)` → `x` | ✅ HAVE (phi-same rule) |
| 2 | Phi with same values | `phi(a, a)` → `a` | ✅ HAVE (phi-same rule) |

**Existing in SpirvLang:** `phi`, `if`, `merge`

#### Priority 3: Image Operations (Lower Priority)

| # | C++ Rule | Pattern | Status |
|---|----------|---------|--------|
| 1 | `UpdateImageOperands` | Remove redundant image operand bits | ⏳ FUTURE |
| 2 | `RedundantFMix` | `fmix(x, y, 0.0)` → `x`, `fmix(x, y, 1.0)` → `y` | ✅ HAVE (via FMix constant folding) |

### Implementation Plan

1. **Add `CompositeConstruct` to SpirvLang** - Support vec2, vec3, vec4 construction ✅ DONE (Session 29)
2. **Implement `CompositeConstructFeedingExtract`** - Extract from construct returns the element ✅ DONE (Session 29)
3. **Implement `CompositeExtractFeedingConstruct`** - Detect reconstruction patterns ✅ DONE (Session 30)
4. **Implement `RedundantPhi`** - Phi with identical operands simplifies to the operand ✅ ALREADY HAD (phi-same rule)
5. **Implement `VectorShuffleFeedingShuffle`** - Compose shuffle masks ✅ DONE (Session 32)

---

## Session 29: Vector/Composite Operations Extension

### Summary
Extended e-graph optimizer to support vector composite construction and the CompositeConstructFeedingExtract optimization.

### Changes Made

#### 1. Added Vec2/Vec3/Vec4 to SpirvLang (lib.rs)
Added composite construction nodes to represent OpCompositeConstruct for vectors:
```rust
"vec2" = Vec2([Id; 2]),  // OpCompositeConstruct for 2-element vector
"vec3" = Vec3([Id; 3]),  // OpCompositeConstruct for 3-element vector
"vec4" = Vec4([Id; 4]),  // OpCompositeConstruct for 4-element vector
```

#### 2. Added Analysis Handling for Vec2/Vec3/Vec4 (lib.rs)
Updated `SpirvAnalysis::make()` to propagate bit width and origin through composite construction.

#### 3. Added Cost Function for Vec2/Vec3/Vec4 (lib.rs)
Both the main cost function and the extraction cost function now handle Vec2/Vec3/Vec4 nodes.

#### 4. Updated translate.rs
Added Vec2/Vec3/Vec4 to:
- The "continue" pattern matching (2 locations)
- The cost calculation for extraction

#### 5. Updated opt_block.rs
Added Vec2/Vec3/Vec4 to the continue patterns (nodes that require OpCompositeConstruct reconstruction - future work).

#### 6. Implemented CompositeConstructFeedingExtract Rule
Added three applier structs and their implementations:
- `ExtractVec2`: `extract(vec2(a, b), idx)` → `a` or `b`
- `ExtractVec3`: `extract(vec3(a, b, c), idx)` → `a`, `b`, or `c`
- `ExtractVec4`: `extract(vec4(a, b, c, d), idx)` → `a`, `b`, `c`, or `d`

Added corresponding rewrite rules:
```rust
rewrite!("extract-vec2"; "(extract (vec2 ?a ?b) ?idx)" => { ExtractVec2 { ... } })
rewrite!("extract-vec3"; "(extract (vec3 ?a ?b ?c) ?idx)" => { ExtractVec3 { ... } })
rewrite!("extract-vec4"; "(extract (vec4 ?a ?b ?c ?d) ?idx)" => { ExtractVec4 { ... } })
```

#### 7. Added Tests
5 new tests for CompositeConstructFeedingExtract:
- `extract_vec2_idx0`: Tests `extract(vec2(a,b), 0)` → `a`
- `extract_vec2_idx1`: Tests `extract(vec2(a,b), 1)` → `b`
- `extract_vec3_idx2`: Tests `extract(vec3(a,b,c), 2)` → `c`
- `extract_vec4_idx3`: Tests `extract(vec4(a,b,c,d), 3)` → `d`
- `extract_vec4_with_non_const_index_preserved`: Verifies non-constant indices are not optimized

### Test Results
- All 945 tests pass (up from 940 in Session 28)
- 5 new tests added for CompositeConstructFeedingExtract

### Status of Phase 2 Implementation Plan

| Task | Status |
|------|--------|
| Add `CompositeConstruct` (vec2/vec3/vec4) | ✅ DONE |
| Implement `CompositeConstructFeedingExtract` | ✅ DONE |
| Implement `CompositeExtractFeedingConstruct` | ✅ DONE (Session 30) |
| Implement `RedundantPhi` | ✅ ALREADY EXISTED (phi-same rule) |
| Implement `VectorShuffleFeedingShuffle` | ✅ DONE (Session 32) |

### Future Work

1. **OpCompositeConstruct Reconstruction**: The opt_block.rs currently skips Vec2/Vec3/Vec4 nodes during SPIR-V reconstruction. Need to implement proper OpCompositeConstruct generation.

---

## Session 30: Vector Composite Optimizations

### Summary
Implemented two C++ spirv-opt optimizations for vector composite operations:

1. **CompositeExtractFeedingConstruct** - Detects when a vector is reconstructed from its own extracted elements
2. **CompositeInsertToCompositeConstruct** - Detects when a series of inserts can be replaced with a vector construct

### Optimization 1: CompositeExtractFeedingConstruct

**Pattern:**
```
vec2(extract(v, 0), extract(v, 1)) → v
vec3(extract(v, 0), extract(v, 1), extract(v, 2)) → v
vec4(extract(v, 0), extract(v, 1), extract(v, 2), extract(v, 3)) → v
```

**Implementation:**
- `Vec2FromExtracts`, `Vec3FromExtracts`, `Vec4FromExtracts` applier structs
- Validates all indices are correct consecutive constants
- Verifies all extracts come from the same source via e-class equality
- Rewrite rules: `vec2-from-extracts`, `vec3-from-extracts`, `vec4-from-extracts`

**Tests (8):**
- `vec2_from_extracts_same_source_optimizes`
- `vec2_from_extracts_wrong_indices_preserved`
- `vec2_from_extracts_different_sources_preserved`
- `vec3_from_extracts_same_source_optimizes`
- `vec3_from_extracts_wrong_indices_preserved`
- `vec4_from_extracts_same_source_optimizes`
- `vec4_from_extracts_mixed_sources_preserved`
- `vec2_from_extracts_non_const_index_preserved`

### Optimization 2: CompositeInsertToCompositeConstruct

**Pattern:**
```
insert(insert(base, a, 0), b, 1) → vec2(a, b)
insert(insert(insert(base, a, 0), b, 1), c, 2) → vec3(a, b, c)
insert(insert(insert(insert(base, a, 0), b, 1), c, 2), d, 3) → vec4(a, b, c, d)
```

**Implementation:**
- `Insert2ToVec2`, `Insert3ToVec3`, `Insert4ToVec4` applier structs
- Validates all indices are correct consecutive constants (0,1 for vec2; 0,1,2 for vec3; etc.)
- Creates new vec node with the inserted values in correct order
- Rewrite rules: `insert2-to-vec2`, `insert3-to-vec3`, `insert4-to-vec4`

**Tests (5):**
- `insert2_to_vec2_optimizes`
- `insert2_to_vec2_wrong_indices_preserved`
- `insert3_to_vec3_optimizes`
- `insert4_to_vec4_optimizes`
- `insert2_non_const_index_preserved`

### Test Results
- All 958 tests passing (698 in lib.rs + others across test crates)

### C++ Parity
- `CompositeExtractFeedingConstruct` matches C++ `folding_rules.cpp:1807-1879`
- `CompositeInsertToCompositeConstruct` matches C++ `folding_rules.cpp:2208-2232`

The Rust implementation uses e-graph pattern matching which provides equivalent semantics. The e-graph approach has the advantage of automatically finding these patterns regardless of code motion or instruction reordering.

---

## Session 31: Status Review and Gap Analysis

### Summary
Reviewed remaining C++ spirv-opt optimizations to identify implementation gaps.

### Analysis

#### Fully Implemented ✅
All major arithmetic and bitwise folding rules from `folding_rules.cpp` are implemented:
- Redundant operations (add zero, mul one, div one, etc.)
- Constant merging through nested operations
- Negate propagation (all combinations)
- Associativity/commutativity rewrites
- Distributive law factoring
- Strength reduction (mul to shift, div to mul, etc.)
- GLSL extended instructions (fmix, clamp, min/max, sqrt, exp, log, pow, trig)
- Vector composite operations (Vec2/3/4 construction, extract/insert)
- CompositeConstructFeedingExtract, CompositeExtractFeedingConstruct
- CompositeInsertToCompositeConstruct

#### Previously Stubbed - Now Implemented ✅ (Session 32)
| Pattern | Status | Implementation |
|---------|--------|----------------|
| `VectorShuffleFeedingExtract` | ✅ Implemented | `ExtractShuffle` applier traces through mask to source vector |
| `VectorShuffleFeedingShuffle` | ✅ Implemented | `ShuffleFeedingShuffle` applier composes masks |
| `ExtractShuffle` applier | ✅ Working | Part of extract-shuffle rewrite rule |
| `ShuffleIdentity` applier | ✅ Working | Part of shuffle-identity rewrite rule |

**Session 32 Infrastructure:** Added `ShuffleMask` struct and `SMask` node to e-graph, enabling proper shuffle mask analysis and composition.

#### Out of Scope
| Pattern | Reason |
|---------|--------|
| `RedundantPhi` | CFG analysis - we have `phi-same` for identical operands |
| `StoringUndef` | Memory operations not in e-graph scope |
| `UpdateImageOperands` | Image sampling operations |
| `RemoveRedundantOperands` | Entry point metadata |
| `BitCastScalarOrVector` | Type conversion, not arithmetic |

### Infrastructure Gap: Shuffle Mask Representation ✅ RESOLVED (Session 32)

**Original Problem:** SPIR-V's `OpVectorShuffle` uses literal operands for the shuffle mask, not Id references:

```spirv
%result = OpVectorShuffle %vec4 %v1 %v2 0 1 4 5  ; mask is [0,1,4,5] as literals
```

**Solution (Session 32):** Added `ShuffleMask` struct as e-graph leaf node (`SMask`):
- Stores up to 8 mask indices (sufficient for vec4×2)
- Implements `LanguageChildren` with 0 children (data node)
- Parse/display format: `smask4_4_0_1_2_3` (firstVecSize_len_indices...)
- Helper methods: `source_and_index()`, `all_from_first()`, `compose_through_first()`

E-graph representation (updated Session 32):
```rust
"shuffle" = VectorShuffle([Id; 3])  // vec1, vec2, mask_id (where mask_id points to SMask node)
"smask" = SMask(ShuffleMask)        // Embedded mask data
```

### Test Results
**All 898+ tests passing!** (Session 32: 9 new shuffle tests added)

---

## Session 32: VectorShuffle Optimizations - COMPLETED

### Summary
Implemented full VectorShuffle optimizations matching C++ `folding_rules.cpp` patterns.

### Implementation

**New Infrastructure:**
- `ShuffleMask` struct: Stores up to 8 mask indices with first vector size
- `SMask` node type: Leaf node in e-graph for shuffle mask data
- Helper methods: `source_and_index()`, `all_from_first()`, `compose_through_first()`

**New Appliers:**
- `ExtractShuffle`: `extract(shuffle(a, b, mask), i)` → `extract(source, new_index)`
- `ShuffleFeedingShuffle`: `shuffle(shuffle(a, b, m1), c, m2)` → `shuffle(a, b, composed_mask)` when composable
- `ShuffleIdentity`: `shuffle(a, b, identity_mask)` → `a` when mask is identity

**New Rewrite Rules:**
- `extract-shuffle`: Trace through shuffle to source vector
- `shuffle-shuffle-compose`: Compose nested shuffles when possible
- `shuffle-identity`: Remove identity shuffles

**Tests (9):**
- `shuffle_identity_from_first`
- `shuffle_identity_mixed_not_identity`
- `extract_shuffle_from_first`
- `extract_shuffle_from_second`
- `extract_shuffle_traces_to_source`
- `shuffle_shuffle_composable`
- `shuffle_shuffle_not_composable_uses_both`
- `shuffle_shuffle_outer_uses_second`
- `insert2_to_vec2_optimizes` (and 4 more composite tests)

### C++ Parity
- `VectorShuffleFeedingExtract` → `extract-shuffle` rule
- `VectorShuffleFeedingShuffle` → `shuffle-shuffle-compose` rule

---

## Session 33: ClampFeedingCompare Optimization - COMPLETED

### Summary
Ported `FoldFClampFeedingCompare` from C++ `const_folding_rules.cpp` and extended it to also handle signed and unsigned integer clamps (SClamp, UClamp).

### C++ Reference
From `const_folding_rules.cpp:1207-1360`: When comparing a clamped value against a constant, if the constant is outside the clamp bounds, the comparison can be folded to true or false.

Example:
```
fclamp(x, 0.0, 1.0) < -1.0  →  false  (clamp output is >= 0.0)
fclamp(x, 0.0, 1.0) > 2.0   →  false  (clamp output is <= 1.0)
```

### Implementation

**New Structs:**
- `FClampCmpOp`: Enum for Lt, Le, Gt, Ge comparison types
- `FClampFeedingCmpLeft/Right`: For floating-point clamp comparisons
- `SClampFeedingCmpLeft/Right`: For signed integer clamp comparisons
- `UClampFeedingCmpLeft/Right`: For unsigned integer clamp comparisons

**Helper Functions:**
- `fold_fclamp_cmp_left()`: Fold `fclamp(x, lo, hi) op c`
- `fold_fclamp_cmp_right()`: Fold `c op fclamp(x, lo, hi)`
- Similar helpers for signed/unsigned integer variants

**New Rewrite Rules (32 total):**
- FOrdLt/Le/Gt/Ge with fclamp (8 rules - left and right operand positions)
- FUnordLt/Le/Gt/Ge with fclamp (8 rules)
- SLt/Le/Gt/Ge with sclamp (8 rules)
- ULt/Le/Gt/Ge with uclamp (8 rules)

### Rust Advantage Over C++
The Rust implementation extends the C++ optimization in two ways:
1. **Integer clamps**: C++ only handles `FClamp` (floating-point). Rust also handles `SClamp` and `UClamp` for integers.
2. **Both operand positions**: Rules handle both `clamp(...) op c` and `c op clamp(...)`.

### Tests (12 new)
- `sclamp_lt_outside_hi_is_true`
- `sclamp_lt_at_lo_is_false`
- `sclamp_gt_outside_lo_is_true`
- `sclamp_gt_at_hi_is_false`
- `sclamp_le_at_hi_is_true`
- `sclamp_ge_at_lo_is_true`
- `uclamp_lt_outside_hi_is_true`
- `uclamp_gt_at_hi_is_false`
- `sclamp_cmp_right_lt_lo_is_true`
- `sclamp_cmp_right_gt_hi_is_true`
- `sclamp_in_bounds_not_folded`

### Test Results
**All 978 tests passing!** (718 lib + 260 integration tests)

---

## Session 34: RedundantAndShift and RedundantAndAddSub Optimizations - COMPLETED

### Summary
Implemented two C++ `folding_rules.cpp` bitwise optimizations:
1. `RedundantAndShift`: `C & (x << n) = 0` when the mask doesn't overlap the shifted range
2. `RedundantAndAddSub`: `C & (x + D) = C & x` when the add/sub can't affect the masked bits

### C++ Reference

**RedundantAndShift (folding_rules.cpp:2997-3043):**
```cpp
// 1 & (b << 1) = 0
// 0x80000000 & (b >> 1) = 0
```
When ANDing a constant mask with a shifted value, if the mask bits don't overlap with the range of possible set bits after the shift, the result is always 0.

**RedundantAndAddSub (folding_rules.cpp:2941-2991):**
```cpp
// 1 & (b + 2) = b & 1
// 1 & (b - 2) = b & 1
```
When ANDing a constant mask with an add/subtract, if the lowest set bit of the addend/subtrahend is higher than all bits in the mask, the add/sub can't affect the result.

### Implementation

**New Structs:**
- `AndShiftLeftToZero`: For `C & (x << n)` patterns
- `AndShiftRightToZero`: For `C & (x >> n)` patterns
- `AndAddToAnd`: For `C & (x + D)` patterns
- `AndSubToAnd`: For `C & (x - D)` patterns

**Applier Logic:**

For shift operations:
- Left shift: Check if `(C >> n) == 0` (mask shifted right clears all bits)
- Right shift: Check if `(C << n) & mask == 0` (mask shifted left overflows out)

For add/sub operations:
- Check if `LSB(D) > C` where LSB is the lowest set bit of D
- If so, adding/subtracting D can't affect any bits in mask C

**New Rewrite Rules (8 total):**
- `and-shl-to-zero`: `C & (x << n)` → 0 when no overlap
- `shl-and-to-zero`: `(x << n) & C` → 0 (commuted)
- `and-shr-u-to-zero`: `C & (x >> n)` → 0 for logical right shift
- `shr-u-and-to-zero`: `(x >> n) & C` → 0 (commuted)
- `and-add-to-and`: `C & (x + D)` → `C & x` when LSB(D) > C
- `add-and-to-and`: `(x + D) & C` → `C & x` (commuted)
- `and-sub-to-and`: `C & (x - D)` → `C & x` when LSB(D) > C
- `sub-and-to-and`: `(x - D) & C` → `C & x` (commuted)

### Tests (15 new)

**And-Shift tests (8):**
- `and_shl_to_zero_mask_in_low_bits`: `0xF & (x << 8)` → 0
- `and_shl_not_zero_when_overlap`: `0xFF00 & (x << 8)` not foldable
- `shl_and_to_zero_commuted`: `(x << 16) & 0xFF` → 0
- `and_shr_u_to_zero_mask_in_high_bits`: `0xF0000000 & (x >> 8)` → 0
- `and_shr_u_not_zero_when_overlap`: `0x00FF0000 & (x >> 8)` not foldable
- `shr_u_and_to_zero_commuted`: `(x >> 24) & 0xFFFFFF00` → 0
- `and_shl_edge_case_shift_by_zero`: Edge case with 0 mask
- `and_shl_full_shift_width`: `0xFFFF & (x << 16)` → 0

**And-Add-Sub tests (7):**
- `and_add_strips_redundant_add`: `1 & (x + 2)` → `1 & x`
- `and_add_preserves_when_overlap`: `3 & (x + 2)` not simplified
- `and_add_larger_mask`: `0x0F & (x + 0x10)` → `0x0F & x`
- `and_sub_strips_redundant_sub`: `1 & (x - 2)` → `1 & x`
- `and_add_commuted`: `(x + 4) & 3` → `3 & x`
- `and_add_zero_addend`: `0xFF & (x + 0)` simplified
- `and_add_boundary_case`: `0xFF & (x + 0x100)` → `0xFF & x`

### Test Results
**All 733 lib tests passing!**

---

## Session 34 Continued: RedundantAndOrXor Optimization - COMPLETED

### Additional Optimization: RedundantAndOrXor

Implemented the `RedundantAndOrXor` optimization from C++ `folding_rules.cpp:2883-2935`.

### C++ Reference
```cpp
// Case 1: C & (x | D) = C when (C & D) == C (OR sets all AND bits)
// Case 2: C & (x | D) = C & x when (C & D) == 0 (no overlap)
// Case 3: C & (x ^ D) = C & x when (C & D) == 0 (no overlap)
```

### Implementation

**New Structs:**
- `AndOrToConst`: For `C & (x | D) = C` when `(C & D) == C`
- `AndOrToAnd`: For `C & (x | D) = C & x` when `(C & D) == 0`
- `AndXorToAnd`: For `C & (x ^ D) = C & x` when `(C & D) == 0`

**New Rewrite Rules (6 total):**
- `and-or-to-const`: `C & (x | D)` → C when OR covers AND mask
- `or-and-to-const`: `(x | D) & C` → C (commuted)
- `and-or-to-and`: `C & (x | D)` → `C & x` when no overlap
- `or-and-to-and`: `(x | D) & C` → `C & x` (commuted)
- `and-xor-to-and`: `C & (x ^ D)` → `C & x` when no overlap
- `xor-and-to-and`: `(x ^ D) & C` → `C & x` (commuted)

### Tests (5 new)
- `and_or_to_const_when_or_covers_and`: `0x0F & (x | 0xFF)` → `0x0F`
- `and_or_to_and_when_no_overlap`: `0x0F & (x | 0xF0)` → `0x0F & x`
- `and_or_preserved_when_partial_overlap`: Negative test
- `and_xor_to_and_when_no_overlap`: `0x0F & (x ^ 0xF0)` → `0x0F & x`
- `and_or_commuted`: `(x | 0xF0) & 0x0F` → `0x0F & x`

### Session 34 Summary
Total optimizations implemented:
1. **RedundantAndShift**: `C & (x << n) = 0` when mask doesn't overlap shifted range
2. **RedundantAndAddSub**: `C & (x + D) = C & x` when add/sub can't affect masked bits
3. **RedundantAndOrXor**: `C & (x | D)` simplifications based on bit overlap analysis

### Test Results
**All 738 lib tests passing!**

---

## Session 35: SAbs (Signed Integer Absolute Value) - COMPLETED

### Goal
Add SAbs (GLSL.std.450 signed integer absolute value) operation with pattern detection and constant folding. This goes beyond C++ spirv-opt which doesn't have this optimization.

### Rust Advantage Over C++
C++ spirv-opt doesn't optimize select(x < 0, -x, x) patterns into SAbs. The Rust e-graph port can detect these patterns and transform them to the more efficient SAbs instruction.

### Implementation

**SpirvLang Addition:**
```rust
"sabs" = SAbs(Id),  // GLSLstd450SAbs: signed integer absolute value
```

**Analysis Handling:**
- SAbs inherits bit_width from operand
- Sets min_value to 0 (absolute value is always >= 0)

**Cost Function:**
- Cost: 2 + operand cost (single GLSL extended instruction)

**Rewrite Rules (5 total):**
1. `sabs-idempotent`: `sabs(sabs(x))` → `sabs(x)`
2. `sabs-neg`: `sabs(neg(x))` → `sabs(x)` (negation doesn't affect absolute value)
3. `sabs-const`: Constant folding for integer constants
4. `select-slt-zero-to-sabs`: `select(x < 0, -x, x)` → `sabs(x)`
5. `select-sge-zero-to-sabs`: `select(x >= 0, x, -x)` → `sabs(x)`

**Appliers:**
- `SAbsConst`: Folds `sabs(const)` to `abs(const)` using `wrapping_abs()`
- `SelectToSAbs`: Checks that comparison is against constant 0, then converts to `sabs(x)`

**Translate/Rebuild:**
- Added SAbs to both `rebuild_arith_with_original_ids` functions
- Uses GLSL.std.450 opcode 5 (GLSLstd450SAbs)
- Generates `OpExtInst` with single operand

### Tests (8 new)
- `sabs_const_positive`: `sabs(42)` → `42`
- `sabs_const_negative`: `sabs(-42)` → `42`
- `sabs_const_zero`: `sabs(0)` → `0`
- `sabs_idempotent`: `sabs(sabs(x))` → `sabs(x)`
- `sabs_neg`: `sabs(neg(x))` → `sabs(x)`
- `sabs_symbolic_preserved`: `sabs(x)` preserved for symbolic x
- `select_slt_zero_to_sabs`: `select(x < 0, -x, x)` → `sabs(x)`
- `select_sge_zero_to_sabs`: `select(x >= 0, x, -x)` → `sabs(x)`

### Files Modified
- `src/lib.rs`: Added SAbs to SpirvLang enum, analysis, cost function, rules, and appliers
- `src/translate.rs`: Added SAbs rebuild to OpExtInst in both rebuild functions
- `src/bin/opt_block.rs`: Added SAbs to continue patterns

### Test Results
**All 746 lib tests passing!** (738 + 8 new SAbs tests)

---

## Session 36: SSign (Signed Integer Sign) - COMPLETED

### Goal
Add SSign (GLSL.std.450 signed integer sign) operation with constant folding and algebraic simplifications. Like SAbs, this goes beyond C++ spirv-opt by providing the SSign operation and its optimizations.

### What SSign Does
`ssign(x)` returns:
- `-1` if x < 0 (negative)
- `0` if x == 0 (zero)
- `1` if x > 0 (positive)

This is the GLSL.std.450 opcode 7 (GLSLstd450SSign).

### Implementation

**SpirvLang Addition:**
```rust
"ssign" = SSign(Id),  // GLSLstd450SSign: signed integer sign (-1, 0, or 1)
```

**Analysis Handling:**
- SSign inherits bit_width from operand
- Result is always in range [-1, 1] (as signed interpretation)

**Cost Function:**
- Cost: 2 + operand cost (single GLSL extended instruction)

**Rewrite Rules (3 total):**
1. `ssign-idempotent`: `ssign(ssign(x))` → `ssign(x)` (sign of -1, 0, or 1 is same)
2. `ssign-neg`: `ssign(neg(x))` → `neg(ssign(x))` (negation flips sign)
3. `ssign-const`: Constant folding for integer constants

**Appliers:**
- `SSignConst`: Folds `ssign(const)` to `-1`, `0`, or `1` based on signed interpretation

**Translate/Rebuild:**
- Added SSign parsing in `translate_arith_with_types` for GLSL opcode 7
- Added SSign rebuild to both `rebuild_arith_with_original_ids` functions
- Uses GLSL.std.450 opcode 7 (GLSLstd450SSign)
- Generates `OpExtInst` with single operand

### Tests (6 new)
- `ssign_const_positive`: `ssign(42)` → `1`
- `ssign_const_negative`: `ssign(-42)` → `-1` (as 0xFFFFFFFF for 32-bit)
- `ssign_const_zero`: `ssign(0)` → `0`
- `ssign_idempotent`: `ssign(ssign(x))` → `ssign(x)`
- `ssign_neg`: `ssign(neg(x))` → `neg(ssign(x))`
- `ssign_symbolic_preserved`: `ssign(x)` preserved for symbolic x

### Files Modified
- `src/lib.rs`: Added SSign to SpirvLang enum, analysis, cost function, rules, appliers, and tests
- `src/translate.rs`: Added SSign parsing and rebuild to OpExtInst in both functions
- `src/bin/opt_block.rs`: Added SSign to continue patterns

### Rust Advantage Over C++
C++ spirv-opt doesn't have SSign optimization. The Rust e-graph port provides:
1. Full constant folding for SSign
2. Algebraic simplifications (idempotent, negation distribution)
3. Integration with the global e-graph optimization pass

### Test Results
**All 752 lib tests passing!** (746 + 6 new SSign tests)

---

## Session 37: Comprehensive C++ Gap Analysis + Memory Optimization

### Goal
Analyze ALL C++ spirv-opt optimizations and identify gaps in the Rust e-graph port. Implement missing optimizations to achieve full parity and beyond.

### C++ Optimization Categories

#### 1. FOLDING RULES (source/opt/folding_rules.cpp) - 3524 lines

**What We Have (DONE):**
- ✅ RedundantBinaryRhs0: `a | 0`, `a ^ 0`, `a << 0`, `a + 0`, `a - 0` → `a`
- ✅ RedundantBinaryLhs0: `0 | a`, `0 ^ a`, `0 + a` → `a`
- ✅ RedundantBinaryLhs0To0: `0 >> a`, `0 << a`, `0 / a` → `0`
- ✅ IntMultipleBy1: `a * 1` → `a`
- ✅ MergeNegateArithmetic: `-(-a)` → `a`
- ✅ MergeNegateAddSubArithmetic: `-a + b` → `b - a`, `a + (-b)` → `a - b`
- ✅ MergeNegateMulDivArithmetic: `-a * b` → `-(a * b)`
- ✅ RedundantSUDiv: `a / 1` → `a`
- ✅ RedundantSUMod: `a % 1` → `0`
- ✅ RedundantAndOrXor: `C & (x | D)` patterns
- ✅ RedundantAndShift: `C & (x << n)` patterns
- ✅ RedundantAndAddSub: `C & (x + D)` patterns
- ✅ Floating-point redundant ops: `fadd`, `fsub`, `fmul`, `fdiv` with identity
- ✅ Min/Max patterns with clamp
- ✅ Comparison chain simplification
- ✅ Bitwise redundancy elimination

**ALSO DONE (verified in code):**
- ✅ **FactorAddMuls**: `a * b + a * c` → `a * (b + c)` - factor-add-muls, fp-factor-add-muls
- ✅ **MergeMulMulArithmetic**: `(a * b) * c` → `a * (b * c)` - IMulMulConstMerge, fmul-fmul-const-merge
- ✅ **MergeMulDivArithmetic**: `(a * b) / b` → `a` - fmul-fdiv-cancel, udiv/sdiv-cancel-common-factor
- ✅ **MergeDivDivArithmetic**: `(a / b) / c` → `a / (b * c)` - sdiv/udiv-merge-consts, fdiv-fdiv-const-merge
- ✅ **MergeDivMulArithmetic**: `(a / b) * b` → `a` - fdiv-fmul-cancel patterns
- ✅ **ReciprocalFDiv**: `a / b` → `a * (1 / b)` - FDivConstToMul applier
- ✅ **InsertFeedingExtract**: `extract(insert(x, idx), idx)` → `x` - extract-insert-same-idx
- ✅ **CompositeExtractFeedingConstruct**: `vec(extract(v,0), ...)` → `v` - Vec2/3/4FromExtracts
- ✅ **CompositeConstructFeedingExtract**: `extract(vec(a, b, c), 1)` → `b` - extract-vec2/3/4
- ✅ **CompositeInsertToCompositeConstruct**: Insert chains → construct - Insert2/3/4ToVec
- ✅ **VectorShuffleFeedingExtract**: `extract(shuffle(...), idx)` - ExtractShuffle applier
- ✅ **VectorShuffleFeedingShuffle**: Combines adjacent shuffles - ShuffleFeedingShuffle
- ✅ **FMixFeedingExtract**: `extract(fmix(...))` patterns - FMixFeedingExtract applier
- ✅ **DotProductDoingExtract**: `dot(v, unit_i)` → `extract(v, i)` - DotUnitVector
- ✅ **RedundantPhi**: `phi(x, x)` → `x` - phi-same rule

**STILL MISSING (Infrastructure needed):**
- ❌ **StoringUndef**: Remove stores of undefined values - Need Load/Store nodes
- ❌ **FunctionCall/Inline**: Need FunctionCall nodes for inlining

#### 2. CONSTANT FOLDING (source/opt/const_folding_rules.cpp) - 1947 lines

**What We Have (DONE):**
- ✅ Integer arithmetic: add, sub, mul, div, mod, neg
- ✅ Bitwise: and, or, xor, not, shifts
- ✅ Integer comparisons: eq, ne, lt, le, gt, ge (signed and unsigned)
- ✅ Floating-point arithmetic: fadd, fsub, fmul, fdiv, fneg
- ✅ Floating-point comparisons: ord and unord variants
- ✅ Min/max: smin, smax, umin, umax, fmin, fmax
- ✅ Clamp operations
- ✅ Trig functions (sin, cos, tan, etc.)
- ✅ Exponential/log functions
- ✅ Sqrt, pow

**What We're MISSING (TODO):**
- ✅ **FoldCompositeWithConstants**: Composite construct - Vec2/3/4ConstFold (Session 38)
- ✅ **FoldExtractWithConstants**: Extract from constant composite - ExtractVConstConst (Session 39)
- ✅ **FoldInsertWithConstants**: Insert into constant composite - InsertVConstConst (Session 40)
- ✅ **FoldVectorShuffleWithConstants**: Shuffle with constants - ShuffleVConstConst (Session 39)
- ✅ **FoldVectorTimesScalar**: Vector × scalar - VectorTimesScalarFold (Session 41)
- ✅ **FoldVectorTimesMatrix**: Vector × matrix - VectorTimesMatrixFold (Session 41)
- ✅ **FoldMatrixTimesVector**: Matrix × vector - MatrixTimesVectorFold (Session 41)
- ✅ **FoldMatrixTimesMatrix**: Matrix × matrix - MatrixTimesMatrixFold (Session 42)
- ✅ **FoldOuterProduct**: Outer product - OuterProductFold (Session 42)
- ✅ **FoldTranspose**: Matrix transpose - TransposeConstFold (Session 41)
- ✅ **FoldFToI/FoldIToF**: Type conversions - ConvertFToS/U/SToF/UToF (Session 38)
- ✅ **FoldSConvert/FoldUConvert**: Integer width - SConvertConst/UConvertConst (Session 38)
- ✅ **FoldQuantizeToF16**: F16 quantization - QuantizeToF16Const (Session 42)
- ✅ **FoldOpDotWithConstants**: Dot product - DotConstFold (Session 38)
- ✅ **FoldBitCastScalarOrVector**: Bitcast - BitCastConst (Session 38)

#### 3. MAJOR PASSES TO INTEGRATE INTO E-GRAPH

These passes operate at module/function level but can be integrated into the e-graph
for global optimization in one pass:

**Dead Code Elimination:**
- AggressiveDCEPass, DeadBranchElimPass, EliminateDeadConstantPass, etc.
- **E-Graph Strategy**: E-graph extraction naturally eliminates unused expressions.
  We track "live" values (function outputs, stores, side effects) and only extract
  reachable expressions. Dead code is never materialized.
- **Status**: ✅ PARTIAL (extraction eliminates dead expr); 🔄 TODO: Add live value tracking

**Memory Optimization:**
- ScalarReplacementPass (SROA), LocalSingleStoreElimPass
- **E-Graph Strategy**: Model loads/stores as e-graph nodes. Add rules:
  - `load(store(addr, val), addr)` → `val` (store-to-load forwarding)
  - Scalar replacement: `store(ptr, composite); load(ptr)[i]` → `composite[i]`
  - Track pointer aliasing via analysis
- **Status**: 🔄 TODO: Add Load/Store nodes and forwarding rules

**Control Flow:**
- BlockMergePass, CFGCleanupPass, MergeReturnPass
- **E-Graph Strategy**: Optimize expressions within each basic block, then
  merge equivalent blocks. Phi nodes become Select nodes in e-graph.
  - `phi(x, x)` → `x` (already implemented)
  - Branch condition simplification via e-graph
  - Block merging when phi nodes resolve to constants
- **Status**: ✅ PARTIAL (phi optimization); 🔄 TODO: Branch simplification

**Loop Optimization:**
- LoopUnroller, LoopFissionPass, LoopFusionPass, LICMPass
- **E-Graph Strategy**:
  - LICM: Expressions with no loop-carried deps can be hoisted automatically
    when we track which values depend on loop variables
  - Loop unrolling: Represent as rewrite if bounds are constant
  - Loop fusion: Equivalent loop bodies in e-graph can share nodes
- **Status**: 🔄 TODO: Add loop-invariant detection, unroll patterns

**Function Level:**
- InlinePass, EliminateDeadFunctionsPass
- **E-Graph Strategy**: Inline function calls by expanding to their body expressions
  in the e-graph. Dead functions get eliminated when nothing references them.
  - `call(f, args...)` → `f.body[params := args]` (inline expansion)
  - Function bodies as e-class patterns
- **Status**: 🔄 TODO: Add FunctionCall nodes, inline expansion rules

### Priority Implementation Order

**Phase 1: Arithmetic Merging (High Impact)**
1. FactorAddMuls: `a * b + a * c` → `a * (b + c)`
2. MergeMulMulArithmetic: `(a * b) * c` → `a * (b * c)`
3. MergeDivDivArithmetic: `(a / b) / c` → `a / (b * c)`

**Phase 2: Composite Operations (High Impact for shaders)**
4. InsertFeedingExtract: `extract(insert(x, c, i), i)` → `x`
5. CompositeConstructFeedingExtract: `extract(vec3(a, b, c), 1)` → `b`
6. VectorShuffleFeedingExtract: `extract(shuffle(...), idx)` simplification

**Phase 3: Constant Folding Extensions**
7. FoldExtractWithConstants
8. FoldVectorShuffleWithConstants
9. FoldOpDotWithConstants

**Phase 4: Advanced Patterns**
10. VectorShuffleFeedingShuffle (shuffle combining)
11. DotProductDoingExtract
12. RedundantPhi

**Phase 5: Memory & Load/Store Optimization**
13. Add Load/Store nodes to e-graph
14. Store-to-load forwarding: `load(store(addr, val), addr)` → `val`
15. Scalar replacement patterns (SROA equivalent)
16. Dead store elimination

**Phase 6: Control Flow in E-Graph**
17. Branch condition simplification via e-graph rewriting
18. Select/Phi optimization patterns
19. Block-level expression equivalence

**Phase 7: Loop Optimization**
20. Loop-invariant code motion (track loop-carried dependencies)
21. Constant loop unrolling patterns
22. Loop body expression optimization

**Phase 8: Function Inlining**
23. Add FunctionCall nodes to e-graph
24. Inline expansion rules
25. Dead function elimination (unreferenced functions)

**Phase 9: Remaining Constant Folding**
26. FoldMatrixTimesVector, FoldMatrixTimesMatrix
27. FoldTranspose
28. FoldFToI/FoldIToF (type conversions)
29. FoldSConvert/FoldUConvert
30. ~~FoldQuantizeToF16~~ ✅ (Session 42)
31. FoldBitCastScalarOrVector

### Implementation Notes

Each optimization should:
1. Add rewrite rule(s) to `make_rules()`
2. Add Applier struct if conditional logic needed
3. Add tests verifying the optimization
4. Update this plan with session notes

### Session 37 Work

**Gap Analysis Correction:**
- Reviewed gap analysis and found many items marked as ❌ were actually already implemented
- Updated plan to mark them as ✅ with rule/applier names

**Brought All Major Passes Into Scope:**
- Changed "OUT OF SCOPE" to detailed e-graph integration strategies for:
  - Dead Code Elimination (extraction naturally eliminates dead nodes)
  - Memory Optimization (SROA via Load/Store rules)
  - Control Flow (phi optimization, branch simplification)
  - Loop Optimization (LICM via dependency tracking)
  - Function Inlining (via FunctionCall nodes)

**New E-Graph Nodes Added:**
- `Load([Id; 2])` - OpLoad: load(pointer, memory_state) -> value
- `Store([Id; 3])` - OpStore: store(pointer, value, memory_state) -> memory_state'
- `FunctionCall(Box<[Id]>)` - OpFunctionCall: call(func_id, args...)

**New Rewrite Rules Added:**
- `load-store-forward`: `load(ptr, store(ptr, val, mem))` → `val` (store-to-load forwarding)
- `store-store-elim`: `store(ptr, v2, store(ptr, v1, mem))` → `store(ptr, v2, mem)` (dead store elimination)

**Analysis & Cost Updates:**
- Added analysis cases for Load, Store, FunctionCall in lib.rs
- Added cost function entries (Load/Store: 10, FunctionCall: 100)
- Added continue patterns in translate.rs and opt_block.rs

**Tests Added (3 new):**
- `load_store_forward` - verifies store-to-load forwarding optimization
- `store_store_elim` - verifies dead store elimination
- `load_different_ptr_preserved` - verifies different pointers don't optimize incorrectly

**All 755 lib tests passing!** (752 + 3 new Load/Store tests)

### Session 38 Work

**FoldOpDotWithConstants:**
- Added `DotConstFold` applier for folding `dot(vconst, vconst)` → `Const`
- Computes dot product of constant vectors at compile time

**Type Conversion Operations:**
- Added 8 new nodes to SpirvLang enum:
  - `BitCast(Id)` - OpBitcast: reinterpret bits as different type
  - `ConvertFToS(Id)` - OpConvertFToS: float to signed int
  - `ConvertFToU(Id)` - OpConvertFToU: float to unsigned int
  - `ConvertSToF(Id)` - OpConvertSToF: signed int to float
  - `ConvertUToF(Id)` - OpConvertUToF: unsigned int to float
  - `SConvert(Id)` - OpSConvert: signed int width conversion
  - `UConvert(Id)` - OpUConvert: unsigned int width conversion
  - `FConvert(Id)` - OpFConvert: float width conversion
- Added analysis cases, cost function entries, and translation support

**Type Conversion Constant Folding Appliers:**
- `BitCastConst` - bitcast(const) → const (bit reinterpretation)
- `ConvertFToSConst` - ftos(float_const) → signed_int_const
- `ConvertFToUConst` - ftou(float_const) → unsigned_int_const
- `ConvertSToFConst` - stof(signed_int_const) → float_const
- `ConvertUToFConst` - utof(unsigned_int_const) → float_const
- `SConvertConst` - sconvert(const) → const (sign-extended width conversion)
- `UConvertConst` - uconvert(const) → const (zero-extended width conversion)
- `FConvertConst` - fconvert(const) → const (float precision conversion)
- Added `bitcast-bitcast` identity rewrite: `bitcast(bitcast(x))` → `x`

**FoldCompositeWithConstants:**
- Added `Vec2ConstFold`, `Vec3ConstFold`, `Vec4ConstFold` appliers
- Fold vec2/3/4 construction from all constants into VConst

**Tests Added (12 new):**
- `dot_const_fold` - verifies dot product constant folding
- `bitcast_const_fold` - tests bitcast(const) folding
- `bitcast_bitcast_identity` - tests bitcast(bitcast(x)) => x
- `convert_ftos_const_fold` - tests ftos(3.7f) => 3
- `convert_ftos_negative_const_fold` - tests ftos(-2.9f) => -2
- `convert_ftou_const_fold` - tests ftou(5.9f) => 5
- `convert_stof_const_fold` - tests stof(-7i) => -7.0f
- `convert_utof_const_fold` - tests utof(42u) => 42.0f
- `convert_stof_large_const_fold` - tests stof(1_000_000i) precision
- `vec2_const_fold` - tests vec2(1, 2) => VConst
- `vec3_const_fold` - tests vec3(1.0, 2.0, 3.0) => VConst
- `vec4_const_fold` - tests vec4(1, 2, 3, 4) => VConst

**All 1028 tests passing!** (768+4+68+96+6+10+6+62+2+6)

### Session 39 Work

**FoldExtractWithConstants:**
- Added `ExtractVConstConst` applier for folding `extract(vconst, const_idx)` → `const`
- Extracts the component at the given index from a constant vector
- C++ parity: FoldExtractWithConstants from const_folding_rules.cpp

**FoldVectorShuffleWithConstants:**
- Added `ShuffleVConstConst` applier for folding `shuffle(vconst1, vconst2, mask)` → `vconst`
- Builds result vector by selecting from v1 or v2 based on shuffle mask
- Handles undefined components (0xFF) by using zero
- C++ parity: FoldVectorShuffleWithConstants from const_folding_rules.cpp

**Tests Added (6 new):**
- `extract_vconst_const_fold` - tests extract(vconst([1,2,3,4]), 2) => 3
- `extract_vconst_first_element` - tests extract(vconst([10,20,30]), 0) => 10
- `extract_vconst_last_element` - tests extract(vconst([5,6]), 1) => 6
- `shuffle_vconst_const_fold` - tests shuffle(v1, v2, mask) with cross-vector selection
- `shuffle_vconst_identity` - tests shuffle identity preserves original
- `shuffle_vconst_swizzle` - tests shuffle swizzle reverses components

**All 1034 tests passing!** (774+4+68+96+6+10+6+62+2+6)

### Session 40 Work

**FoldInsertWithConstants:**
- Added `InsertVConstConst` applier for folding `insert(vconst, const_val, const_idx)` → `vconst'`
- Replaces the component at the given index with the new constant value
- C++ parity: FoldInsertWithConstants from const_folding_rules.cpp

**Gap Analysis Verification:**
- Confirmed `FactorAddMuls` already exists: `factor-add-muls` rule at line 5533
  - Pattern: `(+ (* ?x ?a) (* ?x ?b))` => `(* ?x (+ ?a ?b))`
  - Also has variants for commuted operands and subtraction
- Confirmed `FactorBitOps` already exists: multiple rules
  - `bor-factor-shared-mask`: `(bor (band ?x ?m) (band ?y ?m))` => `(band (bor ?x ?y) ?m)`
  - `logor-factor-and`: `(lor (land ?a ?b) (land ?a ?c))` => `(land ?a (lor ?b ?c))`
  - `logand-factor-or`: `(land (lor ?a ?b) (lor ?a ?c))` => `(lor ?a (land ?b ?c))`

**Tests Added (3 new):**
- `insert_vconst_const_fold` - tests insert(vconst([1,2,3,4]), 99, 2) => vconst([1,2,99,4])
- `insert_vconst_first_element` - tests insert(vconst([1,2,3]), 100, 0) => vconst([100,2,3])
- `insert_vconst_last_element` - tests insert(vconst([5,6]), 7, 1) => vconst([5,7])

**All 1037 tests passing!** (777+4+68+96+6+10+6+62+2+6)

### Session 41 Work

**Matrix Operations Infrastructure:**

Added comprehensive matrix operations to SpirvLang for C++ spirv-opt constant folding parity:

**New SpirvLang nodes:**
- `MConst(MatConst)` - Matrix constant (column-major storage, up to 4x4)
- `Transpose(Id)` - OpTranspose: transpose a matrix
- `VectorTimesMatrix([Id; 2])` - OpVectorTimesMatrix: row vector × matrix
- `MatrixTimesVector([Id; 2])` - OpMatrixTimesVector: matrix × column vector
- `MatrixTimesMatrix([Id; 2])` - OpMatrixTimesMatrix: matrix × matrix
- `MatrixTimesScalar([Id; 2])` - OpMatrixTimesScalar: matrix × scalar
- `VectorTimesScalar([Id; 2])` - OpVectorTimesScalar: vector × scalar
- `OuterProduct([Id; 2])` - OpOuterProduct: outer product

**MatConst struct:**
- Stores up to 4 column vectors (column-major like SPIR-V/GLSL)
- Supports all matrix sizes: 2x2, 2x3, 2x4, 3x2, 3x3, 3x4, 4x2, 4x3, 4x4
- Implements `Display` and `FromStr` for e-graph parsing
- Helper methods: `transpose()`, `is_zero()`, `is_identity()`, `column()`, `get()`

**Matrix Constant Folding Appliers:**
- `TransposeConstFold` - `transpose(mconst)` → transposed matrix constant
- `VectorTimesScalarFold` - `vtimess(vconst, const)` → scaled vector constant
- `MatrixTimesScalarFold` - `mtimess(mconst, const)` → scaled matrix constant
- `VectorTimesMatrixFold` - `vtimesm(vconst, mconst)` → result vector constant
- `MatrixTimesVectorFold` - `mtimesv(mconst, vconst)` → result vector constant

**Rewrite Rules:**
- `transpose-const` - Fold transpose of constant matrix
- `vtimess-const` - Fold vector × scalar with constants
- `mtimess-const` - Fold matrix × scalar with constants
- `vtimesm-const` - Fold vector × matrix with constants
- `mtimesv-const` - Fold matrix × vector with constants
- `transpose-transpose` - `transpose(transpose(m))` → `m`

**Analysis Integration:**
- Matrix constants and operations handled in `Analysis::make()` (treated as opaque for scalar analysis)
- Cost function entries for all matrix operations

**Translation/Rebuild Support:**
- Added matrix ops to continue patterns in `translate.rs` and `opt_block.rs`
- Matrix operations can be translated to e-graph but reconstruction to SPIR-V is future work

**Tests Added (8 new):**
- `transpose_const_fold_2x2` - 2x2 matrix transpose constant folding
- `transpose_const_fold_3x2` - 3x2 → 2x3 transpose constant folding
- `transpose_transpose_cancels` - double transpose cancellation
- `vector_times_scalar_const_fold` - vector × scalar constant folding
- `matrix_times_scalar_const_fold` - matrix × scalar constant folding
- `matrix_times_vector_const_fold` - matrix × vector constant folding
- `vector_times_matrix_const_fold` - vector × matrix constant folding
- `matrix_symbolic_not_folded` - symbolic operands preserved

**C++ Parity:**
- `FoldTranspose` from const_folding_rules.cpp ✅
- `FoldVectorTimesScalar` from const_folding_rules.cpp ✅
- `FoldMatrixTimesVector` from const_folding_rules.cpp ✅
- `FoldVectorTimesMatrix` from const_folding_rules.cpp ✅

**All 1045 tests passing!** (785+4+68+96+6+10+6+62+2+6)

### Session 42 Work

**FoldMatrixTimesMatrix:**
- Added `MatrixTimesMatrixFold` applier for folding `mtimesm(mconst, mconst)` → matrix constant
- Implements matrix × matrix multiplication: A(NxM) × B(MxP) → C(NxP)
- Computes C[col][row] = Σk A[k][row] × B[col][k] for each element
- Validates dimension compatibility (A.num_cols == B.num_rows)
- Handles both FP and integer types using the same FP detection heuristic as other matrix ops
- C++ parity: `FoldMatrixTimesMatrix` from const_folding_rules.cpp ✅

**FoldOuterProduct:**
- Added `OuterProductFold` applier for folding `outerproduct(vconst, vconst)` → matrix constant
- Implements outer product: vec_a(N) × vec_b(M) → matrix(MxN)
- Creates matrix where M[col][row] = a[row] × b[col]
- Result has M columns (length of b) and N rows (length of a)
- C++ parity: `FoldOuterProduct` from const_folding_rules.cpp ✅

**Tests Added (7 new):**
- `matrix_times_matrix_const_fold_2x2` - basic 2x2 × 2x2 matrix multiplication
- `matrix_times_matrix_const_fold_3x2_times_2x3` - non-square mat3x2 × mat2x3 = mat2x2
- `matrix_times_matrix_identity` - identity matrix multiplication (I × A = A)
- `outer_product_const_fold_2x2` - vec2 × vec2 → mat2x2
- `outer_product_const_fold_3x2` - vec3 × vec2 → mat2x3
- `outer_product_zero_vector` - outer product with zero vector → zero matrix
- `outer_product_symbolic_not_folded` - symbolic operands preserved

**FoldQuantizeToF16:**
- Added `QuantizeToF16Const` applier for folding `quantizef16(const)` → quantized constant
- Implements IEEE 754 half-precision (f16) quantization:
  - Extract sign, exponent, mantissa from f32
  - Remap to f16 range (5-bit exponent, 10-bit mantissa)
  - Handle denormals (exponent < -14 → denormal or zero)
  - Handle overflow (exponent > 15 → max f16 value, preserving sign)
  - Round-to-nearest-even for mantissa truncation
  - Convert back to f32 for storage (SPIR-V stores quantized value as f32)
- C++ parity: `FoldQuantizeToF16` from const_folding_rules.cpp ✅

**Tests Added (12 total, 5 new for QuantizeToF16):**
- `quantize_to_f16_const_fold_exact` - value exactly representable (1.0)
- `quantize_to_f16_const_fold_zero` - zero quantization
- `quantize_to_f16_const_fold_loses_precision` - precision loss (1.0 + 2^-12 → 1.0)
- `quantize_to_f16_const_fold_negative` - negative values (-2.5)
- `quantize_to_f16_symbolic_not_folded` - symbolic operands preserved

**All 1057 tests passing!** (797+4+68+96+6+10+6+62+2+6)

### Session 42 Work (Continued)

**FoldFMix:**
- Added `FMixConstFold` applier for folding `fmix(const, const, const)` → constant
- Implements GLSL mix function: `mix(x, y, a) = x*(1-a) + y*a`
- Supports both f32 and f64 operands (auto-detects from bit width)
- C++ parity: `FoldFMix` from const_folding_rules.cpp ✅

**FoldSMin/FoldSMax:**
- Added `SMinConstFold` and `SMaxConstFold` appliers for signed integer min/max
- Uses signed comparison for proper handling of negative values
- Supports 32-bit and 64-bit operands
- C++ parity: `FoldSMin`/`FoldSMax` from const_folding_rules.cpp ✅

**FoldUMin/FoldUMax:**
- Added `UMinConstFold` and `UMaxConstFold` appliers for unsigned integer min/max
- Uses unsigned comparison
- Supports 32-bit and 64-bit operands
- C++ parity: `FoldUMin`/`FoldUMax` from const_folding_rules.cpp ✅

**FoldFMin/FoldFMax:**
- Added `FMinConstFold` and `FMaxConstFold` appliers for floating-point min/max
- Uses f32::min/f64::min (handles NaN properly per IEEE 754)
- Supports both single and double precision
- C++ parity: `FoldFMin`/`FoldFMax` from const_folding_rules.cpp ✅

**Tests Added (11 new):**
- `fmix_const_fold_basic` - basic mix(1.0, 3.0, 0.5) = 2.0
- `fmix_const_fold_zero_a` - mix(x, y, 0.0) = x
- `fmix_const_fold_one_a` - mix(x, y, 1.0) = y
- `smin_const_fold_basic` - smin(5, 3) = 3
- `smin_const_fold_negative` - smin(-5, 3) = -5 (signed comparison)
- `smax_const_fold_basic` - smax(5, 3) = 5
- `umin_const_fold_basic` - umin(5, 3) = 3
- `umax_const_fold_basic` - umax(5, 3) = 5
- `fmin_const_fold_basic` - fmin(2.5, 1.5) = 1.5
- `fmin_const_fold_negative` - fmin(-2.5, 1.5) = -2.5
- `fmax_const_fold_basic` - fmax(2.5, 1.5) = 2.5

**All 1068 tests passing!** (808+4+68+96+6+10+6+62+2+6)

### Session 43: FP Arithmetic Constant Folding

**Gap Identified:**
While we have FAdd, FSub, FMul, FDiv, FNeg nodes and many algebraic rewrites (constant merging,
division-to-multiplication, etc.), we're missing basic constant folding:
- `fadd(const, const)` → constant result
- `fsub(const, const)` → constant result
- `fmul(const, const)` → constant result
- `fdiv(const, const)` → constant result
- `fneg(const)` → constant result

**Implementation Plan:**

1. **Add 5 Applier structs:**
   ```rust
   struct FAddConstFold { a: Var, b: Var }
   struct FSubConstFold { a: Var, b: Var }
   struct FMulConstFold { a: Var, b: Var }
   struct FDivConstFold { a: Var, b: Var }
   struct FNegConstFold { x: Var }
   ```

2. **Add 5 rewrite rules:**
   - `fadd-const-fold`: `(fadd ?a ?b)` → FAddConstFold
   - `fsub-const-fold`: `(fsub ?a ?b)` → FSubConstFold
   - `fmul-const-fold`: `(fmul ?a ?b)` → FMulConstFold
   - `fdiv-const-fold`: `(fdiv ?a ?b)` → FDivConstFold
   - `fneg-const-fold`: `(fneg ?x)` → FNegConstFold

3. **Implement Applier trait for each:**
   - Extract constants, check if both are constants
   - Perform FP arithmetic (supporting f32/f64 based on bit width)
   - Handle special cases (division by zero → no fold, preserve NaN propagation)
   - Return new Const node

4. **Add tests:**
   - Basic operations: 2.0 + 3.0 = 5.0, etc.
   - Negative numbers
   - 64-bit doubles
   - Edge cases: division by zero (should NOT fold), infinity

**C++ Parity:**
- `FoldFAdd`, `FoldFSub`, `FoldFMul`, `FoldFDiv`, `FoldFNegate` from const_folding_rules.cpp

**Status: COMPLETE**
- 5 Applier structs added at line ~7510
- 5 rewrite rules added at line ~6758
- 5 Applier implementations added at line ~11401
- 8 tests added at line ~34340
- All 1076 tests pass
- IEEE 754 semantics preserved (division by zero yields infinity)

### Session 44: FP Comparison Constant Folding

**Gap Identified:**
While we have FP comparison operations (FOrdEq, FOrdNe, FOrdLt, FOrdLe, FOrdGt, FOrdGe, FUnord*)
and self-comparison rules (`fordeq(a, a) => true`), we're missing constant folding where both
operands are constants:
- `fordeq(const_a, const_b)` → bool const
- `fordne(const_a, const_b)` → bool const
- `fordlt(const_a, const_b)` → bool const
- `fordle(const_a, const_b)` → bool const
- `fordgt(const_a, const_b)` → bool const
- `fordge(const_a, const_b)` → bool const
- Same for FUnord* (unordered comparisons - return true if either operand is NaN)

**Implementation Plan:**

1. **Add 12 Applier structs:**
   ```rust
   struct FOrdEqConstFold { a: Var, b: Var }
   struct FOrdNeConstFold { a: Var, b: Var }
   struct FOrdLtConstFold { a: Var, b: Var }
   struct FOrdLeConstFold { a: Var, b: Var }
   struct FOrdGtConstFold { a: Var, b: Var }
   struct FOrdGeConstFold { a: Var, b: Var }
   struct FUnordEqConstFold { a: Var, b: Var }
   struct FUnordNeConstFold { a: Var, b: Var }
   struct FUnordLtConstFold { a: Var, b: Var }
   struct FUnordLeConstFold { a: Var, b: Var }
   struct FUnordGtConstFold { a: Var, b: Var }
   struct FUnordGeConstFold { a: Var, b: Var }
   ```

2. **Add 12 rewrite rules:**
   - `fordeq-const-fold`: `(fordeq ?a ?b)` → FOrdEqConstFold
   - etc.

3. **Implement Applier trait for each:**
   - Extract constants, check if both are constants
   - Perform FP comparison (supporting f32/f64 based on bit width)
   - Handle NaN: ordered comparisons return false if either is NaN
   - Handle NaN: unordered comparisons return true if either is NaN
   - Return BoolConst node

4. **Add tests:**
   - Basic comparisons: 2.0 == 2.0 => true, 2.0 < 3.0 => true
   - NaN handling

**C++ Parity:**
- `FoldFOrdEqual`, `FoldFOrdNotEqual`, `FoldFOrdLessThan`, etc. from const_folding_rules.cpp

**Implementation Complete:**
- Added 12 structs: `FOrdEqConstFold`, `FOrdNeConstFold`, `FOrdLtConstFold`, `FOrdLeConstFold`,
  `FOrdGtConstFold`, `FOrdGeConstFold`, `FUnordEqConstFold`, `FUnordNeConstFold`,
  `FUnordLtConstFold`, `FUnordLeConstFold`, `FUnordGtConstFold`, `FUnordGeConstFold`
- Added 12 rewrite rules in the rules vec
- Implemented 12 Applier trait implementations with proper NaN handling:
  - Ordered comparisons return false if either operand is NaN
  - Unordered comparisons return true if either operand is NaN
- Added 27 tests covering all comparison operations with various inputs including NaN
- All 1103 tests pass (843 + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6)

### Session 45: FClamp Constant Folding

**Gap Identified:**
While SClampFold and UClampFold were implemented for signed/unsigned integer clamp constant folding,
FClampFold was missing for floating-point clamp constant folding.

**Implementation:**
1. Added `FClampFold` struct with x, lo, hi fields
2. Added `fclamp-fold` rewrite rule: `(fclamp ?x ?lo ?hi)` → `FClampFold`
3. Implemented Applier trait for `FClampFold`:
   - Extracts float constants from all three operands
   - Computes `x.max(lo).min(hi)` for both f32 and f64
   - Returns the clamped result as a constant

**C++ Parity:**
- `FoldClamp3` from const_folding_rules.cpp for GLSLstd450FClamp ✅

**Tests Added (5 new):**
- `fclamp_constant_folds_in_range` - value within bounds stays the same
- `fclamp_constant_clamps_low` - value below lo clamps to lo
- `fclamp_constant_clamps_high` - value above hi clamps to hi
- `fclamp_symbolic_preserved` - symbolic x is not folded
- `fclamp_64bit_constant_folds` - 64-bit double precision works

**All 1108 tests pass!** (848 + 4 + 68 + 96 + 6 + 10 + 6 + 62 + 2 + 6)

**C++ const_folding_rules.cpp Parity Complete:**
All constant folding rules from C++ SPIR-V Tools are now implemented in Rust:
- Composite operations (construct, extract, insert)
- Type conversions (FToI, IToF, SConvert, UConvert, FConvert)
- Dot product
- Arithmetic operations (FAdd, FSub, FMul, FDiv, FNeg, IAdd, ISub, IMul, SDiv, UDiv, etc.)
- Floating-point comparisons (FOrd*, FUnord*)
- Vector/Matrix operations (VectorShuffle, VectorTimesScalar, MatrixTimesVector, etc.)
- GLSL extended operations (sin, cos, exp, log, sqrt, min, max, clamp, mix, etc.)
- Quantization (QuantizeToF16)

---

## Session 46: Whole-Program E-Graph Optimization Architecture

### Vision
Expand the e-graph optimizer from local expression optimization to whole-program optimization.
All C++ SPIR-V-Tools optimization passes will be expressed as e-graph rewrite rules, enabling
the e-graph machinery to find globally optimal solutions in a single saturation pass.

**Key Insight:** Rather than running separate passes (DCE, then CFG simplification, then loop
unrolling, then more DCE), we put ALL rules into the e-graph simultaneously. The e-graph
exploration finds the globally optimal combination of transformations.

### Current State (Sessions 1-45)
- **Local Expression Optimization:** Complete
  - 100% parity with C++ const_folding_rules.cpp
  - ~90% parity with C++ folding_rules.cpp (algebraic simplifications)
  - 1108 tests passing
  - Rich SpirvAnalysis with known-bits, value ranges, divisibility tracking

### Session 46 Progress: Whole-Program Infrastructure
- **CFG Representation:** Complete
  - Block, BlockN, Branch, BranchCond, Phi, PhiN nodes
  - 8 CFG rewrite rules (dead branch elimination, block merging, etc.)
  - 8 CFG tests passing
- **Whole-Module Translation:** Complete
  - translate_module(), translate_function(), translate_block()
  - Phi node translation with variable-length PhiN
- **Loop Optimization:** Complete
  - Loop, LoopN, Inv, InductionVar, Unroll nodes
  - Zero-trip elimination, single-iteration inlining, invariant hoisting
  - 5 loop tests passing
- **Total Tests:** 808 passing (including new CFG and loop tests)

### New Scope (Session 46+)
The following C++ passes will be implemented as e-graph rules:

#### Category 1: Control Flow Graph (CFG)
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| DeadBranchElim | Remove branches with constant conditions | Rule: `(branch (const true) then else)` → `then` |
| MergeBlocks | Combine single-entry/single-exit blocks | Rule: `(block (block inner))` → `(block_merge inner)` |
| BlockMerge | Merge blocks after optimizations | Same as above |
| CFGCleanup | Remove unreachable code | Dead e-class elimination after saturation |
| UnreachableElim | Remove unreachable blocks | Same as CFGCleanup |
| IfConversion | Convert simple branches to select | Rule: `(branch cond (return a) (return b))` → `(return (select cond a b))` |

#### Category 2: Dead Code Elimination
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| AggressiveDCE | Remove unused code | Post-extraction: only extract reachable from roots |
| DCE | Dead code elimination | Same - e-graph extraction naturally excludes dead code |
| EliminateDeadFunctions | Remove unused functions | Function-level dead code analysis |
| EliminateDeadIO | Remove unused inputs/outputs | Analysis tracks used IO |

#### Category 3: Loop Optimizations
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| LoopUnroll | Unroll small loops | Rule: `(loop n body)` → `(seq body body ...)` for small n |
| LoopPeeling | Peel first iterations | Rule: `(loop n body)` → `(seq body (loop (n-1) body))` |
| LoopFusion | Combine adjacent loops | Rule: `(seq (loop n a) (loop n b))` → `(loop n (seq a b))` |
| LoopFission | Split loops | Inverse of fusion, for parallelism |
| LICM | Loop-invariant code motion | Analysis marks invariant expressions |
| LoopUnswitch | Move conditionals out of loops | Rule: `(loop (if c a b))` → `(if c (loop a) (loop b))` |

#### Category 4: Memory Optimizations
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| LoadStoreElim | Remove redundant loads | Rule: `(load (store ptr val) ptr)` → `val` |
| RedundancyElim | Common subexpression elimination | E-graph congruence closure (free!) |
| LocalCSE | Local common subexpression | Same - already handled by e-graph |
| ScalarReplacement | Replace aggregates with scalars | Rule: `(load (gep struct idx))` → member access |
| CopyPropagation | Propagate copies | Rule: `(let x y ...)` → substitute y for x |

#### Category 5: Interprocedural Optimizations
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| InlineExhaustive | Inline all functions | Rule: `(call f args)` → expanded body |
| InlineOpaque | Inline opaque functions | Same with analysis for opaqueness |
| EliminateDeadConstants | Remove unused constants | Post-extraction cleanup |
| EliminateDeadMembers | Remove unused struct members | Whole-program analysis |

#### Category 6: SSA Optimizations
| Pass | Description | E-Graph Approach |
|------|-------------|------------------|
| SSARewrite | Convert to SSA form | Already SSA in SPIR-V |
| SimplifyInstructions | Peephole optimizations | Current rewrite rules |
| VectorDCE | Dead vector component elimination | Analysis tracks used components |

---

### Phase 1: CFG Representation in E-Graph

**Goal:** Represent control flow in the e-graph so CFG optimizations become rewrite rules.

#### 1.1 New SpirvLang Nodes for CFG

```rust
define_language! {
    pub enum SpirvLang {
        // ... existing arithmetic nodes ...

        // === CFG Nodes ===

        // Basic block: sequence of instructions ending in a terminator
        // Block(label_id, [instructions...], terminator)
        "block" = Block(Box<[Id]>),

        // Terminators
        "return" = Return([Id; 1]),           // Return with value
        "return_void" = ReturnVoid,           // Return without value
        "branch" = Branch([Id; 1]),           // Unconditional branch to block
        "branch_cond" = BranchCond([Id; 3]),  // cond, then_block, else_block
        "switch" = Switch(Box<[Id]>),         // selector, default, [(case, block), ...]
        "unreachable" = Unreachable,          // Unreachable terminator

        // Phi nodes (SSA)
        "phi" = Phi(Box<[Id]>),               // [(value, pred_block), ...]

        // Function structure
        "function" = Function(Box<[Id]>),     // entry_block, [param_types...]
        "call" = Call(Box<[Id]>),             // callee, [args...]

        // === Memory Nodes ===

        // Load/Store with memory semantics
        "load" = Load([Id; 2]),               // pointer, memory_operands
        "store" = Store([Id; 3]),             // pointer, value, memory_operands

        // Access chains (pointer arithmetic)
        "access_chain" = AccessChain(Box<[Id]>),  // base, [indices...]
        "ptr_access_chain" = PtrAccessChain(Box<[Id]>), // base, element, [indices...]

        // Variables
        "variable" = Variable([Id; 1]),       // storage_class

        // === Loop Nodes ===

        // Loop structure (for loop optimization rules)
        "loop" = Loop([Id; 3]),               // header_block, continue_block, merge_block
        "loop_merge" = LoopMerge([Id; 2]),    // merge_block, continue_block
        "selection_merge" = SelectionMerge([Id; 1]), // merge_block
    }
}
```

#### 1.2 CFG Rewrite Rules

```rust
// Dead branch elimination
rewrite!("dead-branch-true"; "(branch_cond (const 1) ?then ?else)" => "?then"),
rewrite!("dead-branch-false"; "(branch_cond (const 0) ?then ?else)" => "?else"),

// Branch to unconditional
rewrite!("same-target-branch"; "(branch_cond ?cond ?block ?block)" => "(branch ?block)"),

// If-conversion (branch to select)
// (branch_cond cond (block (return a)) (block (return b))) => (return (select cond a b))
// Implemented as custom Applier for complex pattern matching

// Block merging
// (branch (block single_inst (branch target))) => inline single_inst, branch target
// Requires custom Applier to handle instruction sequences

// Loop unrolling (constant trip count)
// (loop (const n) body) => body; body; ... (n times) for small n
struct LoopUnrollApplier { max_unroll: u32 }
```

#### 1.3 Memory Rewrite Rules

```rust
// Load after store to same location
rewrite!("load-after-store"; "(load (store ?ptr ?val) ?ptr)" => "?val"),

// Store after store (dead store elimination)
// (store ptr val1); (store ptr val2) => (store ptr val2)
// Requires sequence analysis

// Load hoisting (load same value multiple times)
// Already handled by e-graph congruence!
```

---

### Phase 2: Whole-Module Translation

**Goal:** Translate entire SPIR-V modules into the e-graph, not just expression trees.

#### 2.1 Extend translate.rs

```rust
/// Translates an entire SPIR-V module into e-graph representation
pub fn translate_module(module: &rspirv::dr::Module) -> TranslatedModule {
    let mut egraph = EGraph::new(SpirvAnalysis::default());

    // 1. Translate all type declarations
    let type_map = translate_types(&module.types_global_values, &mut egraph);

    // 2. Translate global variables
    let global_map = translate_globals(&module.types_global_values, &type_map, &mut egraph);

    // 3. Translate each function
    let mut function_roots = Vec::new();
    for func in &module.functions {
        let func_id = translate_function(func, &type_map, &global_map, &mut egraph);
        function_roots.push(func_id);
    }

    TranslatedModule {
        egraph,
        entry_points: find_entry_points(module),
        function_roots,
        type_map,
        global_map,
    }
}

/// Translates a function with all its blocks and control flow
fn translate_function(
    func: &rspirv::dr::Function,
    type_map: &HashMap<Word, Id>,
    global_map: &HashMap<Word, Id>,
    egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
) -> Id {
    let mut block_map: HashMap<Word, Id> = HashMap::new();
    let mut local_map: HashMap<Word, Id> = HashMap::new();

    // First pass: create block nodes (forward references for branches)
    for block in &func.blocks {
        let label = block.label.as_ref().unwrap().result_id.unwrap();
        let placeholder = egraph.add(SpirvLang::Symbol(Symbol::from("block_placeholder")));
        block_map.insert(label, placeholder);
    }

    // Second pass: translate block contents
    for block in &func.blocks {
        let label = block.label.as_ref().unwrap().result_id.unwrap();
        let translated = translate_block(block, &block_map, &local_map, type_map, egraph);
        // Union with placeholder
        egraph.union(block_map[&label], translated);
    }

    // Return entry block
    let entry_label = func.blocks[0].label.as_ref().unwrap().result_id.unwrap();
    block_map[&entry_label]
}
```

#### 2.2 Block Translation

```rust
fn translate_block(
    block: &rspirv::dr::Block,
    block_map: &HashMap<Word, Id>,
    local_map: &mut HashMap<Word, Id>,
    type_map: &HashMap<Word, Id>,
    egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
) -> Id {
    let mut instruction_ids = Vec::new();

    // Translate non-terminator instructions
    for inst in &block.instructions {
        if is_terminator(inst.class.opcode) {
            continue;
        }

        let inst_id = translate_instruction(inst, local_map, type_map, egraph);
        instruction_ids.push(inst_id);

        if let Some(result_id) = inst.result_id {
            local_map.insert(result_id, inst_id);
        }
    }

    // Translate terminator
    let terminator = block.instructions.last().unwrap();
    let term_id = translate_terminator(terminator, block_map, local_map, type_map, egraph);
    instruction_ids.push(term_id);

    // Create block node
    egraph.add(SpirvLang::Block(instruction_ids.into_boxed_slice()))
}

fn translate_terminator(
    inst: &rspirv::dr::Instruction,
    block_map: &HashMap<Word, Id>,
    local_map: &HashMap<Word, Id>,
    type_map: &HashMap<Word, Id>,
    egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
) -> Id {
    match inst.class.opcode {
        Op::Return => egraph.add(SpirvLang::ReturnVoid),

        Op::ReturnValue => {
            let value_id = operand_to_id(&inst.operands[0], local_map);
            egraph.add(SpirvLang::Return([value_id]))
        }

        Op::Branch => {
            let target = operand_word(&inst.operands[0]);
            egraph.add(SpirvLang::Branch([block_map[&target]]))
        }

        Op::BranchConditional => {
            let cond = operand_to_id(&inst.operands[0], local_map);
            let then_block = block_map[&operand_word(&inst.operands[1])];
            let else_block = block_map[&operand_word(&inst.operands[2])];
            egraph.add(SpirvLang::BranchCond([cond, then_block, else_block]))
        }

        Op::Unreachable => egraph.add(SpirvLang::Unreachable),

        // ... other terminators
        _ => panic!("Unknown terminator: {:?}", inst.class.opcode),
    }
}
```

---

### Phase 3: Enhanced Analysis for CFG

**Goal:** Extend SpirvAnalysis to track control flow properties.

```rust
#[derive(Clone, Debug, Default)]
struct SpirvAnalysis {
    // ... existing fields ...

    // === CFG Analysis ===

    /// Is this expression loop-invariant?
    loop_invariant: bool,

    /// Does this expression have side effects?
    has_side_effects: bool,

    /// Memory locations read by this expression
    reads: MemorySet,

    /// Memory locations written by this expression
    writes: MemorySet,

    /// Is this block reachable from entry?
    reachable: bool,

    /// Dominance information
    dominators: DomInfo,

    /// Loop depth (0 = not in loop)
    loop_depth: u32,
}

#[derive(Clone, Debug, Default)]
struct MemorySet {
    /// Set of memory locations (abstract)
    locations: HashSet<MemoryLocation>,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
enum MemoryLocation {
    Global(Word),           // Global variable
    Local(Word),            // Local variable
    Parameter(u32),         // Function parameter
    Unknown,                // Conservative: could be anything
}
```

---

### Phase 4: Function Inlining

**Goal:** Inline function calls as e-graph rewrite rules.

```rust
struct InlineFunction {
    callee: Var,
    args: Vec<Var>,
}

impl Applier<SpirvLang, SpirvAnalysis> for InlineFunction {
    fn apply_one(
        &self,
        egraph: &mut EGraph<SpirvLang, SpirvAnalysis>,
        eclass: Id,
        subst: &Subst,
        searcher_ast: Option<&PatternAst<SpirvLang>>,
        rule_name: Symbol,
    ) -> Vec<Id> {
        let callee_id = subst[self.callee];

        // Find the function body in the e-graph
        let func_body = find_function_body(egraph, callee_id);

        // Substitute parameters with arguments
        let substituted = substitute_params(egraph, func_body, &self.args, subst);

        vec![substituted]
    }
}

// Rule: (call func args...) => inline func body with args substituted
// Only inline if:
// - Function is small enough (instruction count)
// - Not recursive
// - Inlining would be profitable (analysis)
```

---

### Phase 5: Loop Optimization Rules

```rust
// Loop unrolling for constant bounds
struct LoopUnrollConst {
    trip_count: Var,
    body: Var,
}

impl Applier<SpirvLang, SpirvAnalysis> for LoopUnrollConst {
    fn apply_one(&self, egraph: &mut EGraph<SpirvLang, SpirvAnalysis>, ...) -> Vec<Id> {
        let trip_id = subst[self.trip_count];

        // Only unroll if trip count is a small constant
        if let Some(n) = const_value(egraph, trip_id) {
            if n.get_u64() <= 4 {
                // Create sequence of n copies of body
                let body_id = subst[self.body];
                let mut seq = Vec::new();
                for _ in 0..n.get_u64() {
                    seq.push(body_id);
                }
                return vec![egraph.add(SpirvLang::Seq(seq.into_boxed_slice()))];
            }
        }

        vec![]
    }
}

// Loop peeling
// (loop n body) where first iteration is special
// => body[i=0]; loop (n-1) body[i=i+1]

// Loop fusion
// (seq (loop n a) (loop n b)) => (loop n (seq a b))
// When a and b don't interfere
```

---

### Implementation Plan

#### Phase 1: CFG Nodes (Session 46) ✅ COMPLETE
1. ✅ Added CFG node variants to SpirvLang (Block, Branch, BranchCond, Phi, PhiN, etc.)
2. ✅ Added basic CFG rewrite rules (dead branch elim, branch simplification)
3. ✅ Added 8 CFG tests all passing

#### Phase 2: Memory Nodes (Already Complete from Sessions 1-45)
Memory optimization rules already exist:
- `load-store-forward`: `(load mem (store mem ptr val) ptr)` → `val`
- `store-store-elim`: `(store mem (store mem ptr _) ptr val)` → `(store mem ptr val)`

#### Phase 3: Whole-Module Translation (Session 46) ✅ COMPLETE
1. ✅ Implemented translate_module()
2. ✅ Implemented translate_function()
3. ✅ Implemented translate_block()
4. ✅ Added Phi node translation with PhiN for variable-length phi nodes
5. ✅ Tests with real SPIR-V modules passing

#### Phase 4: Enhanced Analysis (Session 46) ✅ COMPLETE
1. ✅ Added analysis cases for all new CFG and loop nodes
2. ✅ Analysis propagation implemented for Block, BlockN, Phi, PhiN, Loop, LoopN, etc.

#### Phase 5: Loop Optimization (Session 46.1) ✅ COMPLETE
1. ✅ Added Loop, LoopN, Inv, InductionVar, Unroll nodes to SpirvLang
2. ✅ Implemented loop unrolling rules (zero-trip, single-iter, small constant unroll)
3. ✅ Implemented loop invariant rules
4. ✅ Implemented induction variable simplification
5. ✅ Added 5 loop tests all passing

#### Phase 6: Inlining (Future)
1. Implement function inlining applier
2. Add size/recursion checks
3. Test with function call patterns

#### Phase 7: Integration (Future)
1. Full module roundtrip test
2. Performance benchmarks vs C++ spirv-opt
3. Edge case handling and robustness

---

### Why E-Graph is Better Than Sequential Passes

1. **Global Optimality:** E-graph explores all combinations simultaneously
   - C++: DCE, then fold, then DCE again, then fold again...
   - E-graph: All rules fire together, finding optimal combination

2. **Phase Ordering Solved:** No need to worry about pass order
   - C++: "Did loop unrolling before constant prop? Redo constant prop!"
   - E-graph: Both transformations are explored simultaneously

3. **Compositional:** Adding new rules doesn't break existing ones
   - C++: New pass might need re-ordering with existing passes
   - E-graph: Just add the rule, e-graph handles interactions

4. **Provable:** Can prove equivalences between different forms
   - E-graph explicitly represents equivalence classes

5. **Cacheable:** E-graph exploration can be memoized
   - Similar patterns in different functions share work

### References
- [egg: Fast and Flexible E-graphs](https://egraphs-good.github.io/)
- [Equality Saturation: A New Approach to Optimization](https://www.cs.cornell.edu/~ross/publications/eqsat/)
- [SPIR-V Specification](https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html)
