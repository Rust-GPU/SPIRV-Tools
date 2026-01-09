//! Validation context and rule traits.
//!
//! This module provides the shared context passed to all validation rules,
//! similar to how rustc passes `TyCtxt` to various compiler passes.
//!
//! # Architecture
//!
//! The validation system is built around two key abstractions:
//!
//! - [`ValidationContext`]: Holds all shared state needed by validation rules
//! - [`ValidationRule`]: A trait implemented by each validation pass
//!
//! # Adding New Validation Rules
//!
//! To add a new validation rule:
//!
//! 1. Create a struct that implements [`ValidationRule`]
//! 2. Add the rule to the validation pipeline
//!
//! ```ignore
//! pub struct MyNewRule;
//!
//! impl ValidationRule for MyNewRule {
//!     fn name(&self) -> &'static str {
//!         "my-new-rule"
//!     }
//!
//!     fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
//!         for inst in ctx.module().all_inst_iter() {
//!             if ctx.has_capability(Capability::Shader) {
//!                 // validate...
//!             }
//!         }
//!         Ok(())
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};

use rspirv::dr::{Instruction, Module};
use rspirv::spirv::{Capability, ExecutionModel, Op};

use crate::target_env::TargetEnv;
use crate::version::SpirvVersion;

use super::error::ValidationError;
use super::rules::extensions::ExtensionSet;
use super::types::{Id, ResultId, TypeId};
use super::ValidationOptions;

/// Shared context for validation rules.
///
/// This struct holds all the pre-computed state that validation rules need.
/// It's constructed once at the start of validation and passed to all rules.
///
/// Use [`ValidationContextBuilder`] to construct a context.
#[derive(Debug)]
pub struct ValidationContext<'a> {
    /// The parsed SPIR-V module.
    pub module: &'a Module,

    /// Target environment (Vulkan version, OpenCL, etc.).
    pub env: TargetEnv,

    /// Validation options (relaxed rules, limits, etc.).
    pub options: &'a ValidationOptions,

    /// The effective SPIR-V version for this module.
    pub target_version: SpirvVersion,

    /// All result IDs defined in the module (as ResultId).
    pub defined_result_ids: &'a HashSet<ResultId>,

    /// All defined IDs (as Id, for operand validation).
    pub defined_ids: &'a HashSet<Id>,

    /// Map from result ID to the instruction that defines it.
    pub definitions: &'a HashMap<ResultId, Instruction>,

    /// Map from result ID to its opcode.
    pub opcodes: &'a HashMap<ResultId, Op>,

    /// Map from result ID to its result type (for typed instructions).
    pub result_types: &'a HashMap<ResultId, TypeId>,

    /// Declared capabilities in the module (raw from OpCapability).
    pub declared_capabilities: &'a HashSet<Capability>,

    /// Declared extensions in the module.
    pub extensions: &'a ExtensionSet,

    /// Execution models declared via entry points.
    pub entry_models: &'a HashSet<ExecutionModel>,

    /// Member counts for struct types (for decoration validation).
    pub struct_member_counts: &'a HashMap<ResultId, usize>,
}

impl<'a> ValidationContext<'a> {
    // ========================================================================
    // Convenience methods
    // ========================================================================

    /// Checks if a capability is declared.
    #[inline]
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.declared_capabilities.contains(&cap)
    }

    /// Looks up an instruction by its result ID.
    #[inline]
    pub fn get_def(&self, id: ResultId) -> Option<&Instruction> {
        self.definitions.get(&id)
    }

    /// Looks up the opcode for a result ID.
    #[inline]
    pub fn get_opcode(&self, id: ResultId) -> Option<Op> {
        self.opcodes.get(&id).copied()
    }

    /// Checks if this is a Vulkan environment.
    #[inline]
    pub fn is_vulkan(&self) -> bool {
        crate::validation::helpers::is_vulkan_env(self.env)
    }

    /// Checks if a result ID is defined.
    #[inline]
    pub fn is_defined(&self, id: ResultId) -> bool {
        self.defined_result_ids.contains(&id)
    }

    /// Returns the module being validated.
    #[inline]
    pub fn module(&self) -> &'a Module {
        self.module
    }
}

/// A validation rule that can be run against a SPIR-V module.
///
/// Each validation rule implements this trait and is invoked by the
/// validation orchestrator. Rules should be stateless - all state
/// is accessed through the [`ValidationContext`].
pub trait ValidationRule: Sync {
    /// Returns a short, unique name for this rule (for debugging/errors).
    fn name(&self) -> &'static str;

    /// Validates the module according to this rule.
    ///
    /// Returns `Ok(())` if validation passes, or a [`ValidationError`]
    /// describing the first validation failure.
    fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError>;

    /// Returns true if this rule should be skipped based on the context.
    ///
    /// Override this to skip rules that don't apply to certain environments.
    fn should_skip(&self, _ctx: &ValidationContext<'_>) -> bool {
        false
    }
}

/// Runs a sequence of validation rules.
///
/// Runs each rule in order, stopping at the first error.
pub fn run_rules(
    ctx: &ValidationContext<'_>,
    rules: &[&dyn ValidationRule],
) -> Result<(), ValidationError> {
    for rule in rules {
        if !rule.should_skip(ctx) {
            rule.validate(ctx)?;
        }
    }
    Ok(())
}

/// Owned data for constructing a [`ValidationContext`] in tests.
///
/// Since `ValidationContext` holds references, tests need somewhere to own the data.
/// This struct provides owned storage with sensible defaults, and can produce
/// a `ValidationContext` reference via [`as_context()`](Self::as_context).
///
/// # Example
///
/// ```ignore
/// let mut test_ctx = TestContextData::default();
/// test_ctx.options.limits.insert(LIMIT_MAX_SWITCH_BRANCHES, 2);
/// let ctx = test_ctx.as_context();
/// let rule = SwitchBranchLimitRule;
/// rule.validate(&ctx)?;
/// ```
#[derive(Debug)]
pub struct TestContextData {
    /// The parsed SPIR-V module.
    pub module: Module,
    /// Target environment (Vulkan version, OpenCL, etc.).
    pub env: TargetEnv,
    /// Validation options (relaxed rules, limits, etc.).
    pub options: ValidationOptions,
    /// The effective SPIR-V version for this module.
    pub target_version: SpirvVersion,
    /// All result IDs defined in the module.
    pub defined_result_ids: HashSet<ResultId>,
    /// All defined IDs (for operand validation).
    pub defined_ids: HashSet<Id>,
    /// Map from result ID to defining instruction.
    pub definitions: HashMap<ResultId, Instruction>,
    /// Map from result ID to its opcode.
    pub opcodes: HashMap<ResultId, Op>,
    /// Map from result ID to its result type.
    pub result_types: HashMap<ResultId, TypeId>,
    /// Declared capabilities in the module.
    pub declared_capabilities: HashSet<Capability>,
    /// Declared extensions in the module.
    pub extensions: ExtensionSet,
    /// Execution models declared via entry points.
    pub entry_models: HashSet<ExecutionModel>,
    /// Member counts for struct types.
    pub struct_member_counts: HashMap<ResultId, usize>,
}

impl Default for TestContextData {
    fn default() -> Self {
        Self {
            module: Module::new(),
            env: TargetEnv::Vulkan1_0,
            options: ValidationOptions::default(),
            target_version: SpirvVersion::new(1, 0),
            defined_result_ids: HashSet::new(),
            defined_ids: HashSet::new(),
            definitions: HashMap::new(),
            opcodes: HashMap::new(),
            result_types: HashMap::new(),
            declared_capabilities: HashSet::new(),
            extensions: ExtensionSet::default(),
            entry_models: HashSet::new(),
            struct_member_counts: HashMap::new(),
        }
    }
}

impl TestContextData {
    /// Creates a new test context with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a `ValidationContext` referencing this data.
    pub fn as_context(&self) -> ValidationContext<'_> {
        ValidationContext {
            module: &self.module,
            env: self.env,
            options: &self.options,
            target_version: self.target_version,
            defined_result_ids: &self.defined_result_ids,
            defined_ids: &self.defined_ids,
            definitions: &self.definitions,
            opcodes: &self.opcodes,
            result_types: &self.result_types,
            declared_capabilities: &self.declared_capabilities,
            extensions: &self.extensions,
            entry_models: &self.entry_models,
            struct_member_counts: &self.struct_member_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_data_default() {
        let data = TestContextData::default();
        let ctx = data.as_context();
        assert_eq!(ctx.env, TargetEnv::Vulkan1_0);
        assert_eq!(ctx.target_version, SpirvVersion::new(1, 0));
    }
}
