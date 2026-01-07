//! Tests for the SPIR-V optimizer.
//!
//! Tests are organized by category:
//! - `common`: Shared test utilities
//! - `infrastructure`: Error handling, environment variables, overrides
//! - `arithmetic`: Add, sub, mul, div, mod, neg optimizations (integer)
//! - `bitwise`: AND, OR, XOR, NOT, shifts, rotates
//! - `dce`: Dead Code Elimination
//! - `floating_point`: FP arithmetic optimizations (fadd, fsub, fmul, fdiv, fneg)
//! - `select`: Gamma/conditional optimizations

pub(super) mod common;

mod arithmetic;
mod bitwise;
mod dce;
mod floating_point;
mod infrastructure;
mod select;
