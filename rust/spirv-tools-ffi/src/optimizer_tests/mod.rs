//! Tests for the SPIR-V optimizer.
//!
//! Tests are organized by category:
//! - `common`: Shared test utilities
//! - `infrastructure`: Error handling, environment variables, overrides
//! - `arithmetic`: Add, sub, mul, div, mod, neg optimizations (integer)
//! - `bitwise`: AND, OR, XOR, NOT, shifts, rotates
//! - `comparison`: IEqual, INotEqual, SLessThan, ULessThan, FOrdered comparisons
//! - `dce`: Dead Code Elimination
//! - `floating_point`: FP arithmetic optimizations (fadd, fsub, fmul, fdiv, fneg)
//! - `logical`: LogicalAnd, LogicalOr, LogicalNot, Select optimizations
//! - `select`: Gamma/conditional optimizations

pub(super) mod common;

mod arithmetic;
mod bitwise;
mod comparison;
mod dce;
mod floating_point;
mod infrastructure;
mod logical;
mod select;
