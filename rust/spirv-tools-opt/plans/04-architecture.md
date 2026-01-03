# Whole-Program Optimization Architecture

> **Status**: Planned
> **Related**: [Abstract Interpretation](01-abstract-interpretation.md) | [Current Rules](05-current-rules.md)

## Overview

This document describes the architecture for whole-program optimization using e-graphs and RVSDG (Regionalized Value State Dependence Graph) representation.

---

## E-Graph Optimizer Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        SPIR-V Module                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Translation Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  translate  │  │  RVSDG      │  │  Effect Tracking        │ │
│  │  .rs        │  │  Builder    │  │  (Pure vs Effectful)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        E-Graph Core                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  SpirvLang  │  │  Spirv      │  │  Rewrite Rules          │ │
│  │  (Language) │  │  Analysis   │  │  (.egg files)           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Extraction Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Cost       │  │  Extractor  │  │  Rebuild                │ │
│  │  Function   │  │             │  │  (SPIR-V Gen)           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Optimized SPIR-V Module                      │
└─────────────────────────────────────────────────────────────────┘
```

### SpirvLang Enum

The `SpirvLang` enum defines all operations representable in the e-graph:

```rust
define_language! {
    pub enum SpirvLang {
        // Constants
        "const" = Const(ConstValue),
        "const64" = Const64(ConstValue),
        "sym" = Sym(Symbol),
        "arg" = Arg([Id; 2]),

        // Arithmetic
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "neg" = Neg(Id),
        // ... more operations

        // Control Flow (RVSDG)
        "gamma" = Gamma([Id; 3]),      // if-then-else
        "theta" = Theta([Id; 4]),      // loops
        "select" = Select([Id; 3]),    // conditional select
    }
}
```

### SpirvAnalysis

The analysis layer provides rich information for optimization decisions:

```rust
struct SpirvAnalysis {
    // Constant propagation
    constant: ConstLattice,

    // Known-bits analysis
    known_zeros: u64,
    known_ones: u64,

    // Bit width tracking
    bit_width: Option<u32>,

    // Value range
    min_value: Option<u64>,
    max_value: Option<u64>,

    // Divisibility
    known_factor: Option<u64>,

    // Semantic origin
    origin: Origin,
}
```

---

## RVSDG Representation

### Control Flow in E-Graphs

The e-graph uses RVSDG to represent control flow:

#### Gamma (Conditional)

```
Gamma(condition, true_branch, false_branch)
```

Represents if-then-else. The condition determines which branch value is selected.

#### Theta (Loop)

```
Theta(init_values, loop_body, continue_condition, result_selector)
```

Represents loops with explicit loop-carried dependencies.

#### Select (Conditional Value)

```
Select(condition, true_value, false_value)
```

Simpler form for conditional value selection (no side effects).

### Benefits of RVSDG

1. **No CFG complexity** - Control flow is data flow
2. **Explicit dependencies** - All value dependencies are edges
3. **Natural parallelism** - Independent operations visible
4. **Loop optimization** - Loop invariant motion is edge deletion
5. **Dead code elimination** - Unreachable = no edges from root

---

## Optimization Phases

### Phase 1: Translation (SPIR-V → E-Graph)

1. Parse SPIR-V module
2. Build SSA-like value graph
3. Convert to RVSDG structure
4. Insert into e-graph with initial analysis

### Phase 2: Saturation (Optimization)

1. Apply all rewrite rules until saturation
2. Analysis propagates through new nodes
3. E-classes merge equivalent expressions
4. Cost function guides extraction

### Phase 3: Extraction (E-Graph → SPIR-V)

1. Select optimal expression from each e-class
2. Rebuild SPIR-V instructions
3. Generate optimized module

---

## Rule Organization

Rules are organized by category in separate `.egg` files:

| File | Description |
|------|-------------|
| `datatypes.egg` | Core datatype definitions |
| `arithmetic.egg` | Integer arithmetic |
| `bitwise.egg` | Bitwise operations |
| `comparison.egg` | Comparisons |
| `logical.egg` | Logical ops, min/max |
| `floating_point.egg` | FP arithmetic |
| `vector.egg` | Vector operations |
| `matrix.egg` | Matrix operations |
| `glsl.egg` | GLSL extended instructions |
| `rvsdg.egg` | Control flow patterns |
| `constant_folding.egg` | Constant folding |
| `primitives.egg` | Rust primitive rules |

---

## Cost Function

The cost function determines which expression to extract:

```rust
fn cost(node: &SpirvLang) -> f64 {
    match node {
        // Constants are free
        Const(_) | Const64(_) | Sym(_) => 0.0,

        // Simple operations cost 1
        Add(_) | Sub(_) | Mul(_) | Neg(_) => 1.0,
        BitAnd(_) | BitOr(_) | BitXor(_) | BitNot(_) => 1.0,

        // Division/modulo cost more
        SDiv(_) | UDiv(_) | SRem(_) | UMod(_) => 5.0,

        // Complex operations
        FMix(_) | FSmoothStep(_) => 2.0,

        // Clamp is cheaper than nested min/max
        SClamp(_) | UClamp(_) | FClamp(_) => 1.0,
    }
}
```

---

## Future Directions

### 1. Interprocedural Optimization

- Inline small functions into e-graph
- Specialize functions for constant arguments
- Global value numbering across functions

### 2. Memory Optimization

- Track memory dependencies in RVSDG
- Optimize load/store patterns
- Dead store elimination

### 3. Control Flow Optimization

- Loop unrolling with analysis
- Branch elimination with known conditions
- Control flow simplification

### 4. SIMD/Vector Optimization

- Automatic vectorization detection
- Shuffle optimization
- Lane-wise optimization
