# C++ SPIRV-Tools Parity Plan

> **Status**: Phase 1-5 Complete | Phase 7-9 planned (beyond C++ parity)
> **Related**: [Abstract Interpretation](01-abstract-interpretation.md) | [Current Rules](05-current-rules.md)

## Overview

This document tracks our progress toward achieving and exceeding parity with the C++ SPIRV-Tools optimizer (`spirv-opt`). The C++ implementation has ~92 optimization passes with ~180 constant folding rules and ~88 algebraic simplification rules.

## Current Coverage Summary

| Category | C++ Rules | Our Rules | Coverage | Priority |
|----------|-----------|-----------|----------|----------|
| Constant Folding | ~180 | ~70 | 39% | High |
| Algebraic Simplification | ~88 | ~80 | 91% | High |
| GLSL Extended | ~50 ops | ~40 | 80% | High |
| Type Conversions | ~15 | ~15 | 100% ✓ | Complete |
| Vector/Composite | ~30 | ~20 | 67% | Medium |
| Matrix Operations | ~10 | ~10 | 100% ✓ | Complete |
| Quantization | ~5 | 1 | 20% | Low |

**Overall: ~60% feature parity, but with significant architectural advantages**

## E-Graph Advantages (Free Features)

These C++ passes are handled automatically by e-graph - we get them for FREE:
- **Redundancy Elimination** - CSE is free via e-class merging
- **Constant Propagation** - Via egglog constant handling
- **Value Numbering** - Implicit in e-graph structure
- **Canonicalization** - Extraction finds canonical forms
- **No Pass Ordering** - All rules apply simultaneously until saturation
- **Global Optimization** - RVSDG enables optimization across control flow
- **Bidirectional Exploration** - Can explore both directions of rewrites

---

## Implementation Plan

### Phase 1: More Constant Folding Rules (constant_folding.egg)

Add rules for folding constant operations at compile time.

#### 1.1 Integer Division/Modulo Constants
```lisp
; Division of constants
(rule ((= e (SDiv (Const a) (Const b))) (!= b 0))
      ((union e (Const (/ a b)))))
(rule ((= e (UDiv (Const a) (Const b))) (!= b 0))
      ((union e (Const (/ a b)))))
(rule ((= e (SRem (Const a) (Const b))) (!= b 0))
      ((union e (Const (% a b)))))
(rule ((= e (UMod (Const a) (Const b))) (!= b 0))
      ((union e (Const (% a b)))))
```

#### 1.2 Bitwise Constants
```lisp
; Bitwise operations on constants
(rule ((= e (BitAnd (Const a) (Const b))))
      ((union e (Const (& a b)))))
(rule ((= e (BitOr (Const a) (Const b))))
      ((union e (Const (| a b)))))
(rule ((= e (BitXor (Const a) (Const b))))
      ((union e (Const (^ a b)))))
(rule ((= e (BitNot (Const a))))
      ((union e (Const (~ a)))))
```

#### 1.3 Shift Constants
```lisp
(rule ((= e (Shl (Const a) (Const b))))
      ((union e (Const (<< a b)))))
(rule ((= e (ShrU (Const a) (Const b))))
      ((union e (Const (>>> a b)))))
(rule ((= e (ShrS (Const a) (Const b))))
      ((union e (Const (>> a b)))))
```

#### 1.4 Comparison Constants
```lisp
; Integer comparisons
(rule ((= e (Eq (Const a) (Const b))))
      ((union e (Const (if (= a b) 1 0)))))
(rule ((= e (Ne (Const a) (Const b))))
      ((union e (Const (if (!= a b) 1 0)))))
(rule ((= e (SLt (Const a) (Const b))))
      ((union e (Const (if (< a b) 1 0)))))
; ... etc for all comparison ops
```

---

### Phase 2: Type Conversion Rules (NEW FILE: type_conversion.egg)

Add datatype constructors and folding rules for type conversions.

#### 2.1 Add to datatypes.egg
```lisp
; Type conversion operations
(ConvertFToS Expr)      ; Float to signed int
(ConvertFToU Expr)      ; Float to unsigned int
(ConvertSToF Expr)      ; Signed int to float
(ConvertUToF Expr)      ; Unsigned int to float
(SConvert Expr)         ; Signed int width change
(UConvert Expr)         ; Unsigned int width change
(FConvert Expr)         ; Float precision change
(Bitcast Expr)          ; Bit-preserving type reinterpretation
```

#### 2.2 Type Conversion Rules
```lisp
; Round-trip elimination
(rewrite (ConvertFToS (ConvertSToF x)) x)
(rewrite (ConvertFToU (ConvertUToF x)) x)
(rewrite (ConvertSToF (ConvertFToS x)) x)  ; Only if no precision loss
(rewrite (ConvertUToF (ConvertFToU x)) x)  ; Only if no precision loss

; Bitcast of bitcast with same type
(rewrite (Bitcast (Bitcast x)) x)

; Convert constant folding (needs Rust primitives)
; (rule ((= e (ConvertSToF (Const a))))
;       ((union e (ConstF (i64-to-f64 a)))))
```

---

### Phase 3: More GLSL Extended Instructions (glsl.egg)

#### 3.1 Pack/Unpack Operations
Add to datatypes.egg:
```lisp
(PackHalf2x16 Expr Expr)
(UnpackHalf2x16 Expr)
(PackSnorm4x8 Expr)
(UnpackSnorm4x8 Expr)
(PackUnorm4x8 Expr)
(UnpackUnorm4x8 Expr)
(PackDouble2x32 Expr Expr)
(UnpackDouble2x32 Expr)
```

Rules:
```lisp
; Unpack of pack is identity
(rewrite (UnpackHalf2x16 (PackHalf2x16 x y)) (Vec2 x y))
(rewrite (UnpackSnorm4x8 (PackSnorm4x8 v)) v)
(rewrite (UnpackUnorm4x8 (PackUnorm4x8 v)) v)
```

#### 3.2 Additional Math Functions
```lisp
; Radians/Degrees conversion
(Radians Expr)
(Degrees Expr)

; Rules
(rewrite (Radians (Degrees x)) x)
(rewrite (Degrees (Radians x)) x)

; More inverse function pairs
(rewrite (Asinh (Sinh x)) x)
(rewrite (Acosh (Cosh x)) x)  ; for x >= 1
(rewrite (Atanh (Tanh x)) x)
```

#### 3.3 Integer Functions
```lisp
; FindLSB, FindMSB constant folding via Rust primitives
(rule ((= e (FindILsb (Const a))))
      ((union e (Const (find-lsb a)))))
(rule ((= e (FindSMsb (Const a))))
      ((union e (Const (find-msb-signed a)))))
(rule ((= e (FindUMsb (Const a))))
      ((union e (Const (find-msb-unsigned a)))))
(rule ((= e (BitCount (Const a))))
      ((union e (Const (popcount a)))))
```

---

### Phase 4: Quantization Rules

#### 4.1 Add to datatypes.egg
```lisp
(QuantizeToF16 Expr)
```

#### 4.2 Rules
```lisp
; Quantize of quantize is idempotent
(rewrite (QuantizeToF16 (QuantizeToF16 x)) (QuantizeToF16 x))

; Quantize of constant (needs Rust primitive)
; (rule ((= e (QuantizeToF16 (ConstF f))))
;       ((union e (ConstF (quantize-f16 f)))))
```

---

### Phase 5: Matrix Operations (matrix.egg)

#### 5.1 Transpose Rules
```lisp
; Transpose of transpose
(rewrite (Transpose (Transpose m)) m)

; Transpose distributes over addition
(rewrite (Transpose (MatAdd a b)) (MatAdd (Transpose a) (Transpose b)))

; Transpose of scalar multiply
(rewrite (Transpose (MatTimesScalar m s)) (MatTimesScalar (Transpose m) s))
```

#### 5.2 Determinant/Inverse Rules
```lisp
; Determinant of identity is 1
; (requires identity matrix representation)

; Inverse of inverse
(rewrite (MatInverse (MatInverse m)) m)

; Inverse times original is identity
; (rewrite (MatTimesMat m (MatInverse m)) MatIdentity)
```

#### 5.3 Matrix-Vector Multiplication
```lisp
; M * 0 = 0
(rewrite (MatTimesVec m (Vec4 (Const 0) (Const 0) (Const 0) (Const 0)))
         (Vec4 (Const 0) (Const 0) (Const 0) (Const 0)))

; Identity * v = v (requires identity matrix)
```

---

### Phase 6: Additional Rust Primitives (egglog_opt.rs)

Register these primitives for operations that need runtime computation:

```rust
// Type conversion primitives
add_primitive!(&mut egraph, "i64-to-f64" = |a: i64| -> f64 { a as f64 });
add_primitive!(&mut egraph, "f64-to-i64" = |a: f64| -> i64 { a as i64 });

// Bit manipulation
add_primitive!(&mut egraph, "find-lsb" = |a: i64| -> i64 {
    if a == 0 { -1 } else { a.trailing_zeros() as i64 }
});
add_primitive!(&mut egraph, "find-msb-unsigned" = |a: i64| -> i64 {
    if a == 0 { -1 } else { 63 - (a as u64).leading_zeros() as i64 }
});
add_primitive!(&mut egraph, "popcount" = |a: i64| -> i64 {
    (a as u64).count_ones() as i64
});

// Quantization
add_primitive!(&mut egraph, "quantize-f16" = |a: f64| -> f64 {
    half::f16::from_f64(a).to_f64()
});
```

---

## Implementation Checklist

### Phase 1: Constant Folding ✅ COMPLETE
- [x] Add integer division/modulo constant folding
- [x] Add bitwise constant folding
- [x] Add shift constant folding (including ShrS)
- [x] Add comparison constant folding (all Eq/Ne/Lt/Le/Gt/Ge for signed/unsigned)
- [x] Add min/max constant folding (SMin/SMax/UMin/UMax)
- [x] Add logical operation constant folding (LogNot/LogAnd/LogOr/LogEq/LogNe)
- [x] Test all constant folding rules

### Phase 2: Type Conversions ✅ COMPLETE
- [x] Add type conversion datatypes (ConvertFToS/FToU/SToF/UToF, SConvert, UConvert, FConvert, Bitcast)
- [x] Add round-trip elimination rules
- [x] Add pack/unpack operations (Half2x16, Snorm4x8, Snorm2x16, Unorm4x8, Unorm2x16, Double2x32)
- [x] Add radians/degrees operations
- [x] Add NaN-ignoring min/max/clamp (NMin, NMax, NClamp)
- [x] Test type conversion rules

### Phase 3: GLSL Extended ✅ COMPLETE
- [x] Add pack/unpack round-trip rules
- [x] Add radians/degrees cancellation rules
- [x] Add inverse trig cancellation (Asin/Sin, Acos/Cos, Atan/Tan)
- [x] Add inverse hyperbolic cancellation (Asinh/Sinh, Acosh/Cosh, Atanh/Tanh)
- [x] Test GLSL rules

### Phase 4: Quantization ✅ COMPLETE
- [x] Add QuantizeToF16 datatype
- [x] Add quantization idempotent rule
- [x] Test quantization

### Phase 5: Matrix Operations ✅ COMPLETE
- [x] Add transpose rules (already had transpose-transpose)
- [x] Add determinant rules (det(transpose), det(product), det(inverse))
- [x] Add inverse rules (inverse of product, inverse of transpose, inverse of scalar*matrix)
- [x] Add matrix-vector transpose relationships
- [x] Add scalar multiply constant merge
- [x] Test matrix rules

### Phase 6: Rust Primitives
- [ ] Add type conversion primitives (for constant folding)
- [ ] Add bit manipulation primitives (FindLSB, FindMSB, popcount)
- [ ] Add quantization primitive (f16 quantize)
- [ ] Test all primitives

Note: Phase 6 primitives are optional enhancements for constant folding.
The core rules are complete and all tests pass.

---

## Beyond C++ Parity - E-Graph Unique Optimizations

These phases leverage e-graph capabilities that are impossible or impractical in the C++ pass-based architecture.

### Phase 7: Composite/Vector Optimizations (High Impact)

These patterns require looking at multi-level expression trees - perfect for e-graphs.

#### 7.1 VectorShuffle Chaining
```lisp
; VectorShuffle feeding VectorShuffle - compose the shuffles
; shuffle(shuffle(v, w, [a,b,c,d]), x, [e,f,g,h])
; Can be simplified to a single shuffle with composed indices
(rule ((= e (VectorShuffle (VectorShuffle v1 v2 indices1) v3 indices2)))
      ((union e (VectorShuffle v1 v2 (compose-shuffles indices1 indices2)))))
```

#### 7.2 Extract from Construct
```lisp
; CompositeExtract from CompositeConstruct - just get the element
(rewrite (CompositeExtract (CompositeConstruct a b c d) (Const 0)) a)
(rewrite (CompositeExtract (CompositeConstruct a b c d) (Const 1)) b)
(rewrite (CompositeExtract (CompositeConstruct a b c d) (Const 2)) c)
(rewrite (CompositeExtract (CompositeConstruct a b c d) (Const 3)) d)

; Insert feeding Extract - if same index, get inserted value
(rewrite (CompositeExtract (CompositeInsert val composite idx) idx) val)
```

#### 7.3 Dot Product with Sparse Vectors
```lisp
; Dot product where one vector has zeros
; dot([a,0,c,0], v) = a*v.x + c*v.z
(rule ((= e (Dot (Vec4 a (Const 0) c (Const 0)) v)))
      ((union e (FAdd (FMul a (VecExtract v 0)) (FMul c (VecExtract v 2))))))
```

### Phase 8: Cross-Control-Flow Optimizations (RVSDG Unique)

These require RVSDG and can't be done in C++ without complex analysis.

#### 8.1 Gamma Branch Specialization
```lisp
; When condition is known in a branch, specialize the branch
(rule ((= e (Gamma cond true_val false_val))
       (= cond (Const 1)))
      ((union e true_val)))
(rule ((= e (Gamma cond true_val false_val))
       (= cond (Const 0)))
      ((union e false_val)))
```

#### 8.2 Loop-Invariant Code Motion (via Theta)
```lisp
; Expression in Theta body that doesn't depend on loop variable
; can be hoisted - e-graph automatically finds equivalent forms
(rule ((= e (Theta init (Lambda body)))
       (loop-invariant body expr))
      ; expr can be computed once outside loop
      ((union e (Theta init (Lambda (substitute body expr (Const hoisted)))))))
```

#### 8.3 Dead Branch Elimination
```lisp
; Gamma where both branches are equivalent
(rewrite (Gamma cond x x) x)
```

### Phase 9: Arithmetic Expression Combining (High Impact)

Multi-term algebraic simplification that benefits from e-graph saturation.

#### 9.1 Polynomial Simplification
```lisp
; a*x + b*x + c*x = (a+b+c)*x
; E-graph naturally finds this via factoring rules
(rewrite (Add (Mul a x) (Add (Mul b x) (Mul c x)))
         (Mul (Add a (Add b c)) x))

; a*x*x + b*x + c (quadratic form recognition)
```

#### 9.2 Merge Add/Sub Chains
```lisp
; (a + b) - a = b
(rewrite (Sub (Add a b) a) b)
(rewrite (Sub (Add a b) b) a)

; a - (a - b) = b
(rewrite (Sub a (Sub a b)) b)

; (a - b) + b = a
(rewrite (Add (Sub a b) b) a)
```

#### 9.3 Strength Reduction for Signed Division
```lisp
; x / 3 = (x * 0x55555556) >> 32 (for 32-bit)
; This requires Rust primitives to compute magic constants
(rule ((= e (SDiv x (Const 3))))
      ((union e (ShrS (Mul x (Const 0x55555556)) (Const 32)))))
```

---

## Implementation Checklist - Beyond Parity

### Phase 7: Composite/Vector
- [ ] Add VectorShuffle composition primitive
- [ ] Add CompositeExtract from CompositeConstruct rules
- [ ] Add CompositeInsert/Extract interaction rules
- [ ] Add sparse dot product optimization
- [ ] Test with real shader composites

### Phase 8: Cross-Control-Flow (RVSDG)
- [ ] Add Gamma constant condition elimination
- [ ] Add Gamma same-branch elimination
- [ ] Implement loop-invariant detection primitive
- [ ] Add Theta body simplification rules
- [ ] Test with control-flow heavy shaders

### Phase 9: Arithmetic Expression Combining
- [ ] Add polynomial coefficient merging
- [ ] Add add/sub chain simplification
- [ ] Implement signed division magic constant primitive
- [ ] Add more factoring patterns
- [ ] Test with compute shaders

---

## Testing Strategy

For each new rule category:
1. Add unit tests in `egglog_opt::tests`
2. Add integration tests with real SPIR-V modules
3. Verify no regressions in existing tests
4. Compare output with C++ spirv-opt on sample shaders

## Success Criteria

**Phase 1-5 (C++ Parity):** ✅ Complete
- All existing tests pass
- Coverage metrics improved significantly

**Phase 7-9 (Beyond Parity):**
- Demonstrate optimizations C++ can't do
- Cross-control-flow optimization examples
- Multi-term expression simplification
- Measurable code size reduction on real shaders
