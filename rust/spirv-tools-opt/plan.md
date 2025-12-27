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

### Future Improvements

To eventually remove guards entirely, we would need:
1. Identify why the decomposition rules cause incorrect unifications
2. Either fix the underlying rule soundness issue
3. Or use the analysis to make smarter decisions in appliers

The analysis infrastructure now provides the foundation for these improvements.
