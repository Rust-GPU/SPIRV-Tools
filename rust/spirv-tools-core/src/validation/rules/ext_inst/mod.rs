//! Extended instruction validation rules.
//!
//! This module validates SPIR-V extended instructions including:
//!
//! - GLSL.std.450 extended instruction set
//! - OpenCL.std extended instruction set
//! - CLSpv reflection validation
//!
//! The validation is split into submodules for organization:
//!
//! - [`glsl`]: GLSL.std.450 extended instruction validation
//! - [`opencl`]: OpenCL.std extended instruction validation
//! - [`clspv`]: CLSpv reflection extended instruction validation

mod clspv;
mod glsl;
mod opencl;

use crate::validation::context::ValidationRule;

// Re-export all rules for use by the main validation pipeline
pub use clspv::ClspvReflectionRule;
pub use glsl::{
    GlslFloatOpsRule, GlslGeometryOpsRule, GlslIntOpsRule, GlslInterpolateRule, GlslLdexpRule,
    GlslPackUnpackRule, GlslStructOpsRule, GlslTrigOpsRule,
};
pub use opencl::{OpenClFloatOpsRule, OpenClGeometryOpsRule, OpenClIntOpsRule};

// Re-export CLSpv types for external use
pub use clspv::{ClspvInstruction, ClspvInstructionExt};

/// Returns all extended instruction validation rules.
pub fn all_ext_inst_rules() -> Vec<&'static dyn ValidationRule> {
    vec![
        // GLSL.std.450 rules
        &GlslFloatOpsRule,
        &GlslIntOpsRule,
        &GlslTrigOpsRule,
        &GlslPackUnpackRule,
        &GlslGeometryOpsRule,
        &GlslStructOpsRule,
        &GlslLdexpRule,
        &GlslInterpolateRule,
        // OpenCL.std rules
        &OpenClFloatOpsRule,
        &OpenClIntOpsRule,
        &OpenClGeometryOpsRule,
        // NonSemantic.ClspvReflection rules
        &ClspvReflectionRule,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_ext_inst_rules_includes_new_rules() {
        let rules = all_ext_inst_rules();
        let names: Vec<&str> = rules.iter().map(|r| r.name()).collect();

        // GLSL rules
        assert!(
            names.contains(&"glsl-struct-ops"),
            "Missing glsl-struct-ops rule"
        );
        assert!(names.contains(&"glsl-ldexp"), "Missing glsl-ldexp rule");
        assert!(
            names.contains(&"glsl-interpolate"),
            "Missing glsl-interpolate rule"
        );

        // OpenCL rules
        assert!(
            names.contains(&"opencl-float-ops"),
            "Missing opencl-float-ops rule"
        );
        assert!(
            names.contains(&"opencl-int-ops"),
            "Missing opencl-int-ops rule"
        );
        assert!(
            names.contains(&"opencl-geometry-ops"),
            "Missing opencl-geometry-ops rule"
        );

        // CLSpv rules
        assert!(
            names.contains(&"clspv-reflection"),
            "Missing clspv-reflection rule"
        );
    }
}
