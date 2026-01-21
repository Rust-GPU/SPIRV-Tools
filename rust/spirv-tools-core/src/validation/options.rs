//! Validator options and limits.
//!
//! This module provides configuration options for SPIR-V validation, including
//! layout relaxations, limit overrides, and diagnostic settings.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

/// Validator options mirrored from the C++ validator settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationOptions {
    /// Permit relaxed struct store handling.
    pub relax_struct_store: bool,
    /// Permit logical pointer relaxations.
    pub relax_logical_pointer: bool,
    /// Permit relaxed block layout.
    pub relax_block_layout: bool,
    /// Enable uniform buffer standard layout.
    pub uniform_buffer_standard_layout: bool,
    /// Enable scalar block layout.
    pub scalar_block_layout: bool,
    /// Enable workgroup scalar block layout.
    pub workgroup_scalar_block_layout: bool,
    /// Skip block layout validation entirely.
    pub skip_block_layout: bool,
    /// Allow LocalSizeId decoration.
    pub allow_localsizeid: bool,
    /// Allow offset texture operand usage.
    pub allow_offset_texture_operand: bool,
    /// Allow Vulkan 32-bit bitwise operations.
    pub allow_vulkan_32_bit_bitwise: bool,
    /// Enable pre-HLSL legalization relaxations.
    pub before_hlsl_legalization: bool,
    /// Use friendly names for diagnostics.
    pub use_friendly_names: bool,
    /// Validator limit overrides keyed by the limit enum value.
    pub limits: BTreeMap<u32, u32>,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            relax_struct_store: false,
            relax_logical_pointer: false,
            relax_block_layout: false,
            uniform_buffer_standard_layout: false,
            scalar_block_layout: false,
            workgroup_scalar_block_layout: false,
            skip_block_layout: false,
            allow_localsizeid: false,
            allow_offset_texture_operand: false,
            allow_vulkan_32_bit_bitwise: false,
            before_hlsl_legalization: false,
            use_friendly_names: true,
            limits: BTreeMap::new(),
        }
    }
}

impl Hash for ValidationOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.relax_struct_store.hash(state);
        self.relax_logical_pointer.hash(state);
        self.relax_block_layout.hash(state);
        self.uniform_buffer_standard_layout.hash(state);
        self.scalar_block_layout.hash(state);
        self.workgroup_scalar_block_layout.hash(state);
        self.skip_block_layout.hash(state);
        self.allow_localsizeid.hash(state);
        self.allow_offset_texture_operand.hash(state);
        self.allow_vulkan_32_bit_bitwise.hash(state);
        self.before_hlsl_legalization.hash(state);
        self.use_friendly_names.hash(state);
        for (k, v) in &self.limits {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl ValidationOptions {
    /// Returns a copy of the options with the given limit override applied.
    pub fn with_limit(mut self, kind: u32, value: u32) -> Self {
        self.limits.insert(kind, value);
        self
    }
}

/// Limit kind for the maximum number of struct members.
pub const LIMIT_MAX_STRUCT_MEMBERS: u32 = 0;
/// Limit kind for maximum struct nesting depth.
pub const LIMIT_MAX_STRUCT_DEPTH: u32 = 1;
/// Limit kind for maximum local variables.
pub const LIMIT_MAX_LOCAL_VARIABLES: u32 = 2;
/// Limit kind for maximum global variables.
pub const LIMIT_MAX_GLOBAL_VARIABLES: u32 = 3;
/// Limit kind for maximum switch branches.
pub const LIMIT_MAX_SWITCH_BRANCHES: u32 = 4;
/// Limit kind for maximum function arguments.
pub const LIMIT_MAX_FUNCTION_ARGS: u32 = 5;
/// Limit kind for maximum control-flow nesting depth.
pub const LIMIT_MAX_CONTROL_FLOW_NESTING_DEPTH: u32 = 6;
/// Limit kind for maximum access-chain indexes.
pub const LIMIT_MAX_ACCESS_CHAIN_INDEXES: u32 = 7;
/// Limit kind for maximum id bound.
pub const LIMIT_MAX_ID_BOUND: u32 = 8;

/// A simple snapshot of validator limits keyed by the limit enum value.
pub type ValidationLimits = BTreeMap<u32, u32>;
