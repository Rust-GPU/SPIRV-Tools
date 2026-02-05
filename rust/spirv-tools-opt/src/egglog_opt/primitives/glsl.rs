// =============================================================================
// GLSL Transcendental Constant Folding Primitives
// =============================================================================
#![allow(dead_code)]
// These primitives evaluate GLSL math functions on float constants.
// Constants are stored as i64 (bit representation of f64).

/// Compute sin of a float constant.
pub fn float_sin(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sin().to_bits() as i64
}

/// Compute cos of a float constant.
pub fn float_cos(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.cos().to_bits() as i64
}

/// Compute tan of a float constant.
pub fn float_tan(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.tan().to_bits() as i64
}

/// Compute asin of a float constant.
pub fn float_asin(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.asin().to_bits() as i64
}

/// Compute acos of a float constant.
pub fn float_acos(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.acos().to_bits() as i64
}

/// Compute atan of a float constant.
pub fn float_atan(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.atan().to_bits() as i64
}

/// Compute atan2 of two float constants.
pub fn float_atan2(y: i64, x: i64) -> i64 {
    let fy = f64::from_bits(y as u64);
    let fx = f64::from_bits(x as u64);
    fy.atan2(fx).to_bits() as i64
}

/// Compute sinh of a float constant.
pub fn float_sinh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sinh().to_bits() as i64
}

/// Compute cosh of a float constant.
pub fn float_cosh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.cosh().to_bits() as i64
}

/// Compute tanh of a float constant.
pub fn float_tanh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.tanh().to_bits() as i64
}

/// Compute asinh of a float constant.
pub fn float_asinh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.asinh().to_bits() as i64
}

/// Compute acosh of a float constant.
pub fn float_acosh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.acosh().to_bits() as i64
}

/// Compute atanh of a float constant.
pub fn float_atanh(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.atanh().to_bits() as i64
}

/// Compute exp of a float constant.
pub fn float_exp(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.exp().to_bits() as i64
}

/// Compute exp2 of a float constant.
pub fn float_exp2(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.exp2().to_bits() as i64
}

/// Compute log (natural logarithm) of a float constant.
pub fn float_log(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.ln().to_bits() as i64
}

/// Compute log2 of a float constant.
pub fn float_log2(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.log2().to_bits() as i64
}

/// Compute sqrt of a float constant.
pub fn float_sqrt(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.sqrt().to_bits() as i64
}

/// Compute inverse sqrt of a float constant.
pub fn float_inversesqrt(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    (1.0 / f.sqrt()).to_bits() as i64
}

/// Compute pow of two float constants.
pub fn float_pow(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.powf(fy).to_bits() as i64
}

/// Compute floor of a float constant.
pub fn float_floor(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.floor().to_bits() as i64
}

/// Compute ceil of a float constant.
pub fn float_ceil(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.ceil().to_bits() as i64
}

/// Compute round of a float constant.
pub fn float_round(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.round().to_bits() as i64
}

/// Compute trunc of a float constant.
pub fn float_trunc(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.trunc().to_bits() as i64
}

/// Compute abs of a float constant.
pub fn float_abs(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    f.abs().to_bits() as i64
}

/// Compute sign of a float constant (-1, 0, or 1).
pub fn float_sign(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    if f > 0.0 {
        1.0_f64.to_bits() as i64
    } else if f < 0.0 {
        (-1.0_f64).to_bits() as i64
    } else {
        0.0_f64.to_bits() as i64
    }
}

/// Compute fract (fractional part) of a float constant.
pub fn float_fract(x: i64) -> i64 {
    let f = f64::from_bits(x as u64);
    (f - f.floor()).to_bits() as i64
}

/// Compute min of two float constants.
pub fn float_min(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.min(fy).to_bits() as i64
}

/// Compute max of two float constants.
pub fn float_max(x: i64, y: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    fx.max(fy).to_bits() as i64
}

/// Compute clamp of a float constant.
pub fn float_clamp(x: i64, lo: i64, hi: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let flo = f64::from_bits(lo as u64);
    let fhi = f64::from_bits(hi as u64);
    fx.clamp(flo, fhi).to_bits() as i64
}

/// Compute mix (linear interpolation) of float constants.
pub fn float_mix(x: i64, y: i64, a: i64) -> i64 {
    let fx = f64::from_bits(x as u64);
    let fy = f64::from_bits(y as u64);
    let fa = f64::from_bits(a as u64);
    (fx * (1.0 - fa) + fy * fa).to_bits() as i64
}

/// Compute step function.
pub fn float_step(edge: i64, x: i64) -> i64 {
    let fe = f64::from_bits(edge as u64);
    let fx = f64::from_bits(x as u64);
    if fx < fe { 0.0_f64 } else { 1.0_f64 }.to_bits() as i64
}

/// Compute smoothstep function.
pub fn float_smoothstep(edge0: i64, edge1: i64, x: i64) -> i64 {
    let e0 = f64::from_bits(edge0 as u64);
    let e1 = f64::from_bits(edge1 as u64);
    let fx = f64::from_bits(x as u64);
    let t = ((fx - e0) / (e1 - e0)).clamp(0.0, 1.0);
    (t * t * (3.0 - 2.0 * t)).to_bits() as i64
}
