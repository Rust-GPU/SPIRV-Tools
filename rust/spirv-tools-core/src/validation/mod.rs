use std::collections::HashMap;
use std::num::NonZeroU32;

use rspirv::dr::Module;
use thiserror::Error;

use crate::target_env::TargetEnv;

/// Errors that can arise when validating a SPIR-V module.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    /// The module failed to parse before validation could run.
    #[error("failed to parse module: {0}")]
    Parse(String),
    /// The module header is missing.
    #[error("module header is missing")]
    MissingHeader,
    /// The module is missing the required `OpMemoryModel` instruction.
    #[error("OpMemoryModel is required before any function definitions")]
    MissingMemoryModel,
    /// The module declared an invalid id bound (must be greater than zero).
    #[error("declared id bound {bound} is invalid")]
    InvalidIdBound {
        /// The declared id bound from the module header.
        bound: u32,
    },
    /// The module declared an id bound that is exceeded by at least one id.
    #[error("id {id} exceeds declared id bound {bound}")]
    IdExceedsBound {
        /// The offending id value.
        id: u32,
        /// The declared id bound from the module header.
        bound: u32,
    },
    /// Duplicate result ids were found in the module.
    #[error("id {id} is defined more than once")]
    DuplicateResultId {
        /// The result id that was defined multiple times.
        id: u32,
    },
}

/// Validates a SPIR-V module against invariants that can be checked without target-specific
/// knowledge.
pub fn validate_module(words: &[u32], _env: TargetEnv) -> Result<(), ValidationError> {
    let mut loader = rspirv::dr::Loader::new();
    if let Err(error) = rspirv::binary::parse_words(words, &mut loader) {
        return Err(ValidationError::Parse(error.to_string()));
    }
    let module = loader.module();
    validate_header(&module)?;
    validate_id_bound(&module)?;
    validate_memory_model(&module)?;
    Ok(())
}

fn validate_header(module: &Module) -> Result<(), ValidationError> {
    if module.header.is_none() {
        return Err(ValidationError::MissingHeader);
    }
    Ok(())
}

fn validate_memory_model(module: &Module) -> Result<(), ValidationError> {
    if module.memory_model.is_none() {
        return Err(ValidationError::MissingMemoryModel);
    }
    Ok(())
}

fn validate_id_bound(module: &Module) -> Result<(), ValidationError> {
    let header = module
        .header
        .as_ref()
        .ok_or(ValidationError::MissingHeader)?;
    let bound = NonZeroU32::new(header.bound).ok_or(ValidationError::InvalidIdBound {
        bound: header.bound,
    })?;
    let bound_value = bound.get();
    let mut results = HashMap::new();

    let check_id = |id: u32, bound: u32| {
        if id == 0 {
            return None;
        }
        if id >= bound {
            Some(ValidationError::IdExceedsBound { id, bound })
        } else {
            None
        }
    };

    for instruction in module.all_inst_iter() {
        if let Some(id) = instruction.result_id {
            if results.insert(id, ()).is_some() {
                return Err(ValidationError::DuplicateResultId { id });
            }
            if let Some(error) = check_id(id, bound_value) {
                return Err(error);
            }
        }
        if let Some(result_type) = instruction.result_type {
            if let Some(error) = check_id(result_type, bound_value) {
                return Err(error);
            }
        }
        for operand in &instruction.operands {
            if let rspirv::dr::Operand::IdRef(id) = operand {
                if let Some(error) = check_id(*id, bound_value) {
                    return Err(error);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_module, ValidationError};
    use crate::assembly::assemble_text;
    use crate::target_env::TargetEnv;

    #[test]
    fn validate_module_rejects_missing_header() {
        let binary = vec![0x07230203, 0, 0, 0, 0];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::InvalidIdBound { bound: 0 });
    }

    #[test]
    fn validate_module_rejects_ids_beyond_bound() {
        let text = "%void = OpTypeVoid";
        let mut binary = assemble_text(text).expect("assemble");
        // Clamp the declared id bound to 1, which is lower than any type id emitted.
        binary[3] = 1;
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::IdExceedsBound { id: 1, bound: 1 });
    }

    #[test]
    fn validate_module_requires_memory_model() {
        let text = [
            "OpCapability Shader",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }

    #[test]
    fn validate_module_detects_duplicate_result_ids() {
        let text = ["%1 = OpTypeVoid", "%1 = OpTypeInt 32 0"].join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::DuplicateResultId { id: 1 });
    }

    #[test]
    fn validate_module_accepts_valid_binary() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        validate_module(&binary, TargetEnv::Universal1_6).expect("valid module");
    }

    #[test]
    fn validate_module_checks_operand_ids_against_bound() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
            "%fn = OpTypeFunction %void",
            "%main = OpFunction %void None %fn",
            "%entry = OpLabel",
            "OpReturn",
            "OpFunctionEnd",
        ]
        .join("\n");
        let mut binary = assemble_text(&text).expect("assemble");
        // Force a bound that is too small for the function type/result ids.
        binary[3] = 2;
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::IdExceedsBound { id: 2, bound: 2 });
    }
}
