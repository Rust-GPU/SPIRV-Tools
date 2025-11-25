use std::collections::HashMap;
use std::convert::TryFrom;
use std::num::NonZeroU32;

use rspirv::dr::Module;
use thiserror::Error;

use crate::target_env::TargetEnv;

/// A non-zero SPIR-V id.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU32);

impl Id {
    /// Wraps an existing non-zero id.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Attempts to create an `Id` from a raw value, returning `None` if zero.
    pub fn from_raw(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    /// Returns the underlying non-zero id.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<Id> for u32 {
    fn from(id: Id) -> Self {
        id.0.get()
    }
}

impl From<NonZeroU32> for Id {
    fn from(value: NonZeroU32) -> Self {
        Id::new(value)
    }
}

impl TryFrom<u32> for Id {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Id::from_raw(value).ok_or(())
    }
}

impl std::fmt::Display for Id {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A declared upper bound for SPIR-V ids (must be non-zero).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IdBound(NonZeroU32);

impl IdBound {
    /// Wraps an existing non-zero bound.
    pub fn new(value: NonZeroU32) -> Self {
        Self(value)
    }

    /// Attempts to create an id bound from a raw value, returning `None` if zero.
    pub fn from_raw(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    /// Returns the underlying non-zero bound value.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

impl From<IdBound> for u32 {
    fn from(bound: IdBound) -> Self {
        bound.0.get()
    }
}

impl From<NonZeroU32> for IdBound {
    fn from(value: NonZeroU32) -> Self {
        IdBound::new(value)
    }
}

impl TryFrom<u32> for IdBound {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        IdBound::from_raw(value).ok_or(())
    }
}

impl std::fmt::Display for IdBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Errors that can arise when validating a SPIR-V module.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
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
    /// The module declared more than one memory model instruction.
    #[error("multiple OpMemoryModel instructions are not allowed")]
    DuplicateMemoryModel,
    /// A function definition appeared before the memory model.
    #[error("OpMemoryModel must appear before any function definitions")]
    FunctionBeforeMemoryModel,
    /// An instruction was found before the required memory model declaration.
    #[error("instruction {opcode:?} cannot appear before OpMemoryModel")]
    InstructionBeforeMemoryModel {
        /// The opcode that violated the ordering.
        opcode: rspirv::spirv::Op,
    },
    /// Global instructions are out of the required logical layout order.
    #[error("instruction {opcode:?} appears out of order in the logical layout")]
    LayoutOutOfOrder {
        /// The opcode that violated the ordering.
        opcode: rspirv::spirv::Op,
    },
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
        id: Id,
        /// The declared id bound from the module header.
        bound: IdBound,
    },
    /// Duplicate result ids were found in the module.
    #[error("id {id} is defined more than once")]
    DuplicateResultId {
        /// The result id that was defined multiple times.
        id: Id,
    },
}

/// Validates a SPIR-V module against invariants that can be checked without target-specific
/// knowledge.
pub fn validate_module(words: &[u32], _env: TargetEnv) -> Result<(), ValidationError> {
    run_layout_check(words)?;
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

fn run_layout_check(words: &[u32]) -> Result<(), ValidationError> {
    struct LayoutChecker {
        memory_models: usize,
        current_section: usize,
        inside_function: usize,
        pre_memory_model_violation: Option<ValidationError>,
    }

    impl LayoutChecker {
        fn new() -> Self {
            Self {
                memory_models: 0,
                current_section: 0,
                inside_function: 0,
                pre_memory_model_violation: None,
            }
        }
    }

    impl rspirv::binary::Consumer for LayoutChecker {
        fn initialize(&mut self) -> rspirv::binary::ParseAction {
            rspirv::binary::ParseAction::Continue
        }

        fn finalize(&mut self) -> rspirv::binary::ParseAction {
            if self.memory_models == 0 {
                if let Some(error) = &self.pre_memory_model_violation {
                    return rspirv::binary::ParseAction::Error(Box::new(error.clone()));
                }
                return rspirv::binary::ParseAction::Error(Box::new(
                    ValidationError::MissingMemoryModel,
                ));
            }
            rspirv::binary::ParseAction::Continue
        }

        fn consume_header(&mut self, _: rspirv::dr::ModuleHeader) -> rspirv::binary::ParseAction {
            rspirv::binary::ParseAction::Continue
        }

        fn consume_instruction(
            &mut self,
            inst: rspirv::dr::Instruction,
        ) -> rspirv::binary::ParseAction {
            if self.inside_function > 0 {
                match inst.class.opcode {
                    rspirv::spirv::Op::MemoryModel => {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::FunctionBeforeMemoryModel,
                        ));
                    }
                    rspirv::spirv::Op::Function => self.inside_function += 1,
                    rspirv::spirv::Op::FunctionEnd => self.inside_function -= 1,
                    _ => {}
                }
                return rspirv::binary::ParseAction::Continue;
            }

            match inst.class.opcode {
                rspirv::spirv::Op::MemoryModel => {
                    if self.current_section > SECTION_MEMORY_MODEL {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder {
                                opcode: rspirv::spirv::Op::MemoryModel,
                            },
                        ));
                    }
                    self.memory_models += 1;
                    if self.memory_models > 1 {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::DuplicateMemoryModel,
                        ));
                    }
                    self.current_section = self.current_section.max(SECTION_MEMORY_MODEL);
                }
                rspirv::spirv::Op::Function => {
                    if self.memory_models == 0 {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::FunctionBeforeMemoryModel,
                        ));
                    }
                    self.inside_function = 1;
                    self.current_section = self.current_section.max(SECTION_FUNCTIONS);
                }
                opcode => {
                    let section = section_index(opcode);
                    if section < self.current_section {
                        return rspirv::binary::ParseAction::Error(Box::new(
                            ValidationError::LayoutOutOfOrder { opcode },
                        ));
                    }
                    self.current_section = self.current_section.max(section);
                    if section > SECTION_MEMORY_MODEL && self.memory_models == 0 {
                        if self.pre_memory_model_violation.is_none() {
                            self.pre_memory_model_violation =
                                Some(ValidationError::InstructionBeforeMemoryModel { opcode });
                        }
                        return rspirv::binary::ParseAction::Continue;
                    }
                }
            }
            rspirv::binary::ParseAction::Continue
        }
    }

    let mut checker = LayoutChecker::new();
    match rspirv::binary::parse_words(words, &mut checker) {
        Ok(()) => Ok(()),
        Err(rspirv::binary::ParseState::ConsumerError(err)) => {
            if let Some(validation) = err.downcast_ref::<ValidationError>() {
                Err(validation.clone())
            } else {
                Err(ValidationError::Parse(err.to_string()))
            }
        }
        Err(other) => Err(ValidationError::Parse(other.to_string())),
    }
}

const SECTION_CAPABILITIES: usize = 0;
const SECTION_MEMORY_MODEL: usize = 1;
const SECTION_ENTRY_AND_MODES: usize = 2;
const SECTION_DEBUG: usize = 3;
const SECTION_NAMES: usize = 4;
const SECTION_ANNOTATIONS: usize = 5;
const SECTION_TYPES_GLOBALS: usize = 6;
const SECTION_FUNCTIONS: usize = 7;

fn section_index(opcode: rspirv::spirv::Op) -> usize {
    use rspirv::spirv::Op::*;
    match opcode {
        Capability | Extension | ExtInstImport => SECTION_CAPABILITIES,
        MemoryModel => SECTION_MEMORY_MODEL,
        EntryPoint | ExecutionMode | ExecutionModeId => SECTION_ENTRY_AND_MODES,
        String | SourceExtension | Source | SourceContinued | ModuleProcessed => SECTION_DEBUG,
        Name | MemberName => SECTION_NAMES,
        Decorate | DecorateId | MemberDecorate | DecorateString | MemberDecorateString
        | GroupDecorate | GroupMemberDecorate => SECTION_ANNOTATIONS,
        Function => SECTION_FUNCTIONS,
        _ => SECTION_TYPES_GLOBALS,
    }
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
    let bound = IdBound::from_raw(header.bound).ok_or(ValidationError::InvalidIdBound {
        bound: header.bound,
    })?;
    let mut results = HashMap::new();

    let check_id = |id: u32, bound: IdBound| {
        Id::try_from(id).ok().and_then(|id| {
            if id.get() >= bound.get() {
                Some(ValidationError::IdExceedsBound { id, bound })
            } else {
                None
            }
        })
    };

    for instruction in module.all_inst_iter() {
        if let Some(id) = instruction.result_id {
            if let Some(valid_id) = Id::from_raw(id) {
                if results.insert(valid_id, ()).is_some() {
                    return Err(ValidationError::DuplicateResultId { id: valid_id });
                }
            }
            if let Some(error) = check_id(id, bound) {
                return Err(error);
            }
        }
        if let Some(result_type) = instruction.result_type {
            if let Some(error) = check_id(result_type, bound) {
                return Err(error);
            }
        }
        for operand in &instruction.operands {
            if let rspirv::dr::Operand::IdRef(id) = operand {
                if let Some(error) = check_id(*id, bound) {
                    return Err(error);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_module, Id, IdBound, ValidationError};
    use crate::assembly::assemble_text;
    use crate::target_env::TargetEnv;

    fn op(word_count: u16, opcode: u16) -> u32 {
        ((word_count as u32) << 16) | opcode as u32
    }

    #[test]
    fn validate_module_rejects_missing_header() {
        let binary = vec![0x07230203, 0, 0, 0, 0];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }

    #[test]
    fn validate_module_rejects_ids_beyond_bound() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%void = OpTypeVoid",
        ]
        .join("\n");
        let mut binary = assemble_text(&text).expect("assemble");
        // Clamp the declared id bound to 1, which is lower than any type id emitted.
        binary[3] = 1;
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::IdExceedsBound {
                id: Id::from_raw(1).unwrap(),
                bound: IdBound::from_raw(1).unwrap()
            }
        );
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
        assert_eq!(
            error,
            ValidationError::InstructionBeforeMemoryModel {
                opcode: rspirv::spirv::Op::TypeVoid,
            }
        );
    }

    #[test]
    fn validate_module_detects_duplicate_result_ids() {
        let text = [
            "OpCapability Shader",
            "OpMemoryModel Logical GLSL450",
            "%1 = OpTypeVoid",
            "%1 = OpTypeInt 32 0",
        ]
        .join("\n");
        let binary = assemble_text(&text).expect("assemble");
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(
            error,
            ValidationError::DuplicateResultId {
                id: Id::from_raw(1).unwrap()
            }
        );
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
        assert_eq!(
            error,
            ValidationError::IdExceedsBound {
                id: Id::from_raw(2).unwrap(),
                bound: IdBound::from_raw(2).unwrap(),
            }
        );
    }

    #[test]
    fn validate_module_rejects_memory_model_after_function() {
        // Manually build a module where OpMemoryModel appears after the function body.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version 1.0
            0,          // generator
            5,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
            op(5, 54), // OpFunction %3
            1,
            3,
            0,
            2,
            op(2, 248), // OpLabel %4
            4,
            op(1, 253), // OpReturn
            op(1, 56),  // OpFunctionEnd
            op(3, 14),  // OpMemoryModel Logical GLSL450 (misordered)
            0,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::FunctionBeforeMemoryModel);
    }

    #[test]
    fn validate_module_rejects_duplicate_memory_model() {
        // Build a minimal module with two memory model instructions.
        let binary = vec![
            0x07230203,
            0x00010000,
            0,
            5,
            0,
            op(2, 17), // OpCapability Shader
            1,
            op(3, 14), // OpMemoryModel Logical GLSL450
            0,
            1,
            op(3, 14), // Duplicate OpMemoryModel
            0,
            1,
            op(2, 19), // OpTypeVoid %1
            1,
            op(3, 33), // OpTypeFunction %2 %1
            2,
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::DuplicateMemoryModel);
    }

    #[test]
    fn validate_module_reports_missing_memory_model_without_other_globals() {
        // A module that declares only capabilities should still fail for a missing memory model.
        let binary = vec![
            0x07230203, // magic
            0x00010000, // version
            0,          // generator
            1,          // bound
            0,          // schema
            op(2, 17),  // OpCapability Shader
            1,
        ];
        let error = validate_module(&binary, TargetEnv::Universal1_6).unwrap_err();
        assert_eq!(error, ValidationError::MissingMemoryModel);
    }
}
