//! Tests for the SPIR-V optimizer.
//!
//! Tests are organized by category:
//! - `common`: Shared test utilities
//! - `infrastructure`: Error handling, environment variables, overrides
//! - `arithmetic`: Add, sub, mul, div, mod, neg optimizations
//! - `bitwise`: AND, OR, XOR, NOT, shifts, rotates
//! - `select`: Gamma/conditional optimizations

pub(super) mod common;

mod arithmetic;
mod bitwise;
mod infrastructure;
mod select;
