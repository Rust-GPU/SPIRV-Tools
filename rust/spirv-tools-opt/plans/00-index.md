# SPIRV-Tools Rust E-Graph Optimizer - Planning Documents

This directory contains the planning and design documents for the Rust-based SPIR-V optimizer using e-graphs (egglog).

## Document Index

| File | Description | Status |
|------|-------------|--------|
| [01-abstract-interpretation.md](01-abstract-interpretation.md) | Abstract interpretation analysis design | Planned |
| [02-cpp-parity.md](02-cpp-parity.md) | C++ SPIRV-Tools parity analysis & plan | In Progress |
| [03-session-log.md](03-session-log.md) | Historical session log | Archive |
| [04-architecture.md](04-architecture.md) | Whole-program optimization architecture | Planned |
| [05-current-rules.md](05-current-rules.md) | Summary of current optimization rules | Reference |

## Quick Links

### Current Work
- **[C++ Parity Plan](02-cpp-parity.md#implementation-plan)** - What we need to achieve parity with C++ spirv-opt

### Architecture
- **[E-Graph Design](01-abstract-interpretation.md)** - How the optimizer works
- **[RVSDG Representation](04-architecture.md)** - Control flow in e-graphs

### Reference
- **[Rule Files](../src/rules/)** - The actual egglog rule files
- **[Current Coverage](05-current-rules.md)** - What optimizations we support

## Directory Structure

```
spirv-tools-opt/
├── plans/                    # This directory
│   ├── 00-index.md          # This file
│   ├── 01-abstract-interpretation.md
│   ├── 02-cpp-parity.md
│   ├── 03-session-log.md
│   ├── 04-architecture.md
│   └── 05-current-rules.md
├── src/
│   ├── rules/               # Egglog rule files
│   │   ├── datatypes.egg    # Core datatype definitions
│   │   ├── arithmetic.egg   # Integer arithmetic rules
│   │   ├── bitwise.egg      # Bitwise operation rules
│   │   ├── comparison.egg   # Comparison rules
│   │   ├── logical.egg      # Logical operation rules
│   │   ├── floating_point.egg # FP arithmetic rules
│   │   ├── vector.egg       # Vector operation rules
│   │   ├── matrix.egg       # Matrix operation rules
│   │   ├── glsl.egg         # GLSL extended instruction rules
│   │   ├── rvsdg.egg        # Control flow (Gamma/Theta) rules
│   │   ├── constant_folding.egg # Constant folding rules
│   │   └── primitives.egg   # Rules using Rust primitives
│   ├── egglog_opt.rs        # E-graph optimizer implementation
│   ├── translate.rs         # SPIR-V to e-graph translation
│   └── lib.rs               # Main library entry
└── tests/                   # Integration tests
```

## Goals

1. **Parity with C++ spirv-opt** - Match the optimization capabilities of the C++ implementation
2. **Global optimization** - Leverage e-graphs for whole-program optimization
3. **Correctness** - All transformations must be semantics-preserving
4. **Performance** - Fast compilation with good optimization quality
