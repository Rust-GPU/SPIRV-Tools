//! SPIR-V validation rules organized by category.
//!
//! This module contains the validation logic for SPIR-V modules, organized
//! into logical submodules for maintainability and extensibility.
//!
//! # Adding New Validation Rules
//!
//! To add a new validation rule:
//!
//! 1. Identify the appropriate submodule (or create a new one)
//! 2. Create a struct implementing [`ValidationRule`]:
//!    ```ignore
//!    pub struct MyNewRule;
//!
//!    impl ValidationRule for MyNewRule {
//!        fn name(&self) -> &'static str {
//!            "my-new-rule"
//!        }
//!
//!        fn validate(&self, ctx: &ValidationContext<'_>) -> Result<(), ValidationError> {
//!            // validation logic
//!            Ok(())
//!        }
//!    }
//!    ```
//! 3. Add the rule to the appropriate `all_*_rules()` function
//!
//! # Module Organization
//!
//! - [`capabilities`]: Capability validation and dependency checking
//! - [`extensions`]: Extension validation and version requirements
//! - [`limits`]: Resource limit enforcement (id bounds, struct depth, etc.)
//! - [`storage_classes`]: Storage class validation rules
//! - [`builtins`]: Built-in variable validation rules
//! - [`interpolation`]: Interpolation decoration validation rules
//! - [`decorations`]: Decoration validation rules
//! - [`layout`]: Block and memory layout validation (legacy)
//! - [`cfg`]: Control flow graph validation
//! - [`memory`]: Memory model validation rules
//! - [`pointers`]: Pointer and store validation rules
//! - [`vulkan`]: Vulkan-specific validation rules
//! - [`entry_points`]: Entry point interface validation rules
//! - [`execution_modes`]: Execution mode validation rules
//! - [`types`]: Type and ID validation rules
//! - [`block_layout`]: Block layout validation rules
//! - [`arithmetics`]: Arithmetic instruction validation rules
//! - [`adjacency`]: Instruction placement and adjacency validation rules
//! - [`literals`]: Literal number encoding validation rules
//! - [`derivatives`]: Derivative instruction validation rules
//! - [`barriers`]: Barrier instruction validation rules
//! - [`atomics`]: Atomic instruction validation rules
//! - [`mesh_shading`]: Mesh shading instruction validation rules
//! - [`debug`]: Debug instruction validation rules
//! - [`memory_semantics`]: Memory semantics validation rules
//! - [`scopes`]: Scope validation rules

pub mod adjacency;
pub mod arithmetics;
pub mod atomics;
pub mod barriers;
pub mod bitwise;
pub mod block_layout;
pub mod builtins;
pub mod capabilities;
pub mod cfg;
pub mod composites;
pub mod constants;
pub mod conversion;
pub mod debug;
pub mod decorations;
pub mod derivatives;
pub mod entry_points;
pub mod execution_modes;
pub mod extensions;
pub mod functions;
pub mod image;
pub mod interpolation;
pub mod layout;
pub mod limits;
pub mod literals;
pub mod logicals;
pub mod memory;
pub mod memory_semantics;
pub mod mesh_shading;
pub mod non_uniform;
pub mod pointers;
pub mod ray_tracing;
pub mod scopes;
pub mod storage_classes;
pub mod types;
pub mod vulkan;
