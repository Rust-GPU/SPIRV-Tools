# Current Optimization Rules Summary

> **Status**: Reference
> **Related**: [C++ Parity](02-cpp-parity.md)

This document summarizes the optimization rules currently implemented in the Rust e-graph optimizer.

## Rule Files

All rules are in `src/rules/*.egg`:

| File | Lines | Description |
|------|-------|-------------|
| `datatypes.egg` | ~300 | Core datatype definitions (Expr, Effect, ExprList) |
| `arithmetic.egg` | ~250 | Integer arithmetic optimizations |
| `bitwise.egg` | ~80 | Bitwise operation optimizations |
| `comparison.egg` | ~30 | Comparison simplifications |
| `logical.egg` | ~65 | Logical operations and min/max |
| `floating_point.egg` | ~120 | Floating-point optimizations |
| `vector.egg` | ~180 | Vector operations |
| `matrix.egg` | ~30 | Matrix operations |
| `glsl.egg` | ~230 | GLSL extended instructions |
| `rvsdg.egg` | ~200 | Control flow (Gamma/Theta) |
| `constant_folding.egg` | ~50 | Constant folding |
| `primitives.egg` | ~80 | Rules using Rust primitives |

## Datatypes (datatypes.egg)

### Expr Sort (Pure Values)
```
Arithmetic: Add, Sub, Mul, Neg, SDiv, UDiv, SRem, SMod, UMod
Bitwise: Shl, ShrS, ShrU, BitAnd, BitOr, BitXor, BitNot, BitReverse, RotL, RotR
Comparison: Eq, Ne, SLt, SLe, SGt, SGe, ULt, ULe, UGt, UGe
Min/Max: SMin, SMax, UMin, UMax
Logical: LogNot, LogAnd, LogOr, LogEq, LogNe
Control: Gamma, Select, If, Theta, LoopVar, LoopInvariant
Float: FAdd, FSub, FMul, FDiv, FNeg, FRem, FMin, FMax, FAbs, FFloor, FCeil, FRound, FTrunc
Float Cmp: FOrdEq, FOrdNe, FOrdLt, FOrdLe, FOrdGt, FOrdGe, FUnord*
FP Pred: IsNan, IsInf
Vector: Vec2, Vec3, Vec4, VecExtract, VecInsert, VecAdd, VecSub, VecMul, VecDiv, VecNeg, VecF*, Dot, VecTimesScalar
Matrix: MatTimesScalar, MatTimesVec, VecTimesMat, MatTimesMat, Transpose, OuterProduct, Determinant, MatInverse
GLSL: FMix, SmoothStep, FClamp, SClamp, UClamp, Cross, Normalize, Length, Distance, Reflect, Refract, FaceForward
GLSL Math: Sqrt, InverseSqrt, Exp, Exp2, Log, Log2, Pow, Sin, Cos, Tan, Asin, Acos, Atan, Atan2, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Fma, Fract, Modf, ModfStruct, Ldexp, Frexp, FrexpStruct, Step, Sign, FSign, SAbs, FindILsb, FindSMsb, FindUMsb, BitCount
Constants: Const, Const64, Sym, Arg
```

### Effect Sort (Side Effects)
```
Pure, Return, ReturnValue, Seq, EffGamma, Unreachable
```

## Arithmetic Rules (arithmetic.egg)

### Identity Rules
- `x + 0 = x`, `x - 0 = x`, `x * 1 = x`, `x * 0 = 0`

### Negation Rules
- `0 - x = -x`, `--x = x`, `-1 * x = -x`
- `-(x + y) = -x + -y`, `-(x - y) = y - x`
- `x + (-x) = 0`, `x - x = 0`

### Algebraic Cancellation
- `(x + y) - x = y`, `(x + y) - y = x`
- `(x - y) + y = x`, `x - (x + y) = -y`
- `(a - b) - (c - b) = a - c`

### Strength Reduction (Multiply)
- `x * 2 = x << 1`, `x * 4 = x << 2`, ... `x * 256 = x << 8`
- `x * 3 = (x << 1) + x`
- `x * 5 = (x << 2) + x`
- `x * 7 = (x << 3) - x`
- `x * 9 = (x << 3) + x`
- `x * 15 = (x << 4) - x`

### Strength Reduction (Divide)
- `x / 2 = x >> 1` (unsigned), `x / 4 = x >> 2`, etc.
- `x % 2 = x & 1`, `x % 4 = x & 3`, etc.

### Division/Modulo
- `(x * a) / a = x`, `x / 1 = x`, `x / x = 1`
- `0 / x = 0`, `0 % x = 0`, `x % 1 = 0`, `x % x = 0`

### Factoring
- `(x * a) + (x * b) = x * (a + b)`
- `(x * a) - (x * b) = x * (a - b)`
- `x + x = x * 2`

### Shift Combining
- `(x << a) << b = x << (a + b)`
- `(x >> a) >> b = x >> (a + b)`

## Bitwise Rules (bitwise.egg)

### Identity/Annihilator
- `x | 0 = x`, `x & ~0 = x`, `x ^ 0 = x`
- `x | ~0 = ~0`, `x & 0 = 0`

### Idempotent
- `x | x = x`, `x & x = x`, `x ^ x = 0`

### Complement
- `x | ~x = ~0`, `x & ~x = 0`, `~~x = x`

### De Morgan's Laws
- `~(x & y) = ~x | ~y`, `~(x | y) = ~x & ~y`

### Absorption
- `x | (x & y) = x`, `x & (x | y) = x`

### Shift Identities
- `x << 0 = x`, `x >> 0 = x`

## Floating-Point Rules (floating_point.egg)

### Identity
- `x * 1 = x`, `x + 0 = x`, `x / 1 = x`

### Negation
- `--x = x`, `-(-x * y) = x * y`
- `(-x) * (-y) = x * y`

### Subtraction
- `x - x = 0`, `x - y = x + (-y)`

### Min/Max
- `fmin(x, x) = x`, `fmax(x, x) = x`
- `fmin(-a, -b) = -fmax(a, b)`

### Floor/Ceil
- `floor(floor(x)) = floor(x)`
- `floor(-x) = -ceil(x)`

### Comparison Negation
- `!(a == b) = a != b` (ordered/unordered pairs)

## GLSL Rules (glsl.egg)

### FMix (Linear Interpolation)
- `fmix(a, a, t) = a`

### Clamp
- `fclamp(x, a, a) = a`
- `fclamp(fclamp(x, lo, hi), lo, hi) = fclamp(x, lo, hi)`
- `fclamp(x, lo, hi) = fmax(lo, fmin(x, hi))`

### Sqrt/InverseSqrt
- `x * inversesqrt(x) = sqrt(x)`
- `1 / sqrt(x) = inversesqrt(x)`
- `sqrt(x) * sqrt(x) = x`

### Exp/Log
- `log(exp(x)) = x`, `exp(log(x)) = x`
- `log2(exp2(x)) = x`, `exp2(log2(x)) = x`
- `exp(a + b) = exp(a) * exp(b)`
- `log(a * b) = log(a) + log(b)`
- `log(a / b) = log(a) - log(b)`

### Trigonometric
- `sin(-x) = -sin(x)`, `cos(-x) = cos(x)`
- `tan(x) = sin(x) / cos(x)`
- `sinh(-x) = -sinh(x)`, `cosh(-x) = cosh(x)`

### FMA
- `a * b + c = fma(a, b, c)`
- `fma(a, b, 0) = a * b`
- `fma(1, b, c) = b + c`

### Abs/Sign
- `abs(abs(x)) = abs(x)`
- `abs(-x) = abs(x)`
- `sign(sign(x)) = sign(x)`
- `abs(x) * sign(x) = x`

### Normalize/Fract
- `normalize(normalize(x)) = normalize(x)`
- `fract(x) = x - floor(x)`
- `fract(fract(x)) = fract(x)`

## Vector Rules (vector.egg)

### Extract/Insert
- `extract(vec2(a, b), 0) = a`, `extract(vec2(a, b), 1) = b`
- `extract(insert(v, x, i), i) = x`
- `extract(insert(v, x, i), j) = extract(v, j)` when i != j

### Identity
- `v + vec(0, 0, ...) = v`
- `v * vec(1, 1, ...) = v`
- `--v = v`

### Dot Product
- `dot(a, b) = dot(b, a)`
- `dot(unit_x, v) = extract(v, 0)`

### Cross Product
- `cross(a, b) = -cross(b, a)`
- `cross(-a, b) = -cross(a, b)`

## Rust Primitives (primitives.egg)

### Registered Primitives
- `bitrev32`, `bitrev64` - Bit reversal
- `bits-disjoint` - Check if masks don't overlap
- `shl-clears-mask`, `shr-clears-mask` - Check if shift clears mask
- `mask-superset` - Check if one mask contains another

### Primitive Rules
- `BitReverse(Const a) = Const(bitrev32(a))`
- `mask & (x + const) = mask & x` when bits disjoint
- `mask & (x << shift) = 0` when shift clears mask

## Statistics

| Category | Rules |
|----------|-------|
| Arithmetic | ~60 |
| Bitwise | ~30 |
| Comparison | ~15 |
| Logical | ~25 |
| Floating-Point | ~50 |
| Vector | ~90 |
| Matrix | ~15 |
| GLSL | ~80 |
| RVSDG | ~40 |
| Constant Folding | ~20 |
| Primitives | ~15 |
| **Total** | **~440** |
