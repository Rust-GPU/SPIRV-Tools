//! Friendly name resolution for validation error messages.
//!
//! This module provides the `FriendlyNames` type which collects `OpName` and
//! `OpMemberName` debug instructions to produce more readable error messages.

use std::collections::HashMap;

use rspirv::dr::Module;

use super::error::ValidationError;
use super::options::ValidationOptions;
use super::types::MemberIndex;

/// User-facing names collected from debug instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriendlyNames {
    id_names: HashMap<u32, String>,
    member_names: HashMap<(u32, MemberIndex), String>,
}

impl FriendlyNames {
    /// Constructs a friendly-name table from raw id/member maps.
    pub fn from_parts(
        id_names: HashMap<u32, String>,
        member_names: HashMap<(u32, MemberIndex), String>,
    ) -> Self {
        FriendlyNames {
            id_names,
            member_names,
        }
    }

    /// Returns any `OpName`-provided name for the given result id.
    pub fn id(&self, id: u32) -> Option<&str> {
        self.id_names.get(&id).map(String::as_str)
    }

    /// Returns any `OpMemberName`-provided name for the given struct/member pair.
    pub fn member(&self, struct_id: u32, member: MemberIndex) -> Option<&str> {
        self.member_names
            .get(&(struct_id, member))
            .map(String::as_str)
    }

    /// Formats an id with a friendly suffix when available (e.g., `%5 (foo)`).
    pub fn format_id(&self, id: u32) -> String {
        if let Some(name) = self.id(id) {
            format!("%{id} ({name})")
        } else {
            format!("%{id}")
        }
    }

    /// Formats a struct member with a friendly suffix when available.
    pub fn format_member(&self, struct_id: u32, member: MemberIndex) -> String {
        if let Some(name) = self.member(struct_id, member) {
            format!("%{struct_id}.{member} ({name})")
        } else {
            format!("%{struct_id}.{member}")
        }
    }

    /// Accesses the raw id→name table.
    pub fn id_names(&self) -> &HashMap<u32, String> {
        &self.id_names
    }

    /// Accesses the raw (struct id, member)→name table.
    pub fn member_names(&self) -> &HashMap<(u32, MemberIndex), String> {
        &self.member_names
    }
}

/// Formats a validation error, appending friendly names when provided.
pub fn format_validation_error(error: &ValidationError, names: Option<&FriendlyNames>) -> String {
    match (error, names) {
        (ValidationError::ExecutionModeWithoutEntryPoint { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::InvalidEntryPointTarget { target, .. }, Some(names)) => {
            names.format_id((*target).into())
        }
        (
            ValidationError::EntryPointInterfaceStorageClassInvalid {
                entry_point,
                interface,
                ..
            },
            Some(names),
        ) => format!(
            "{} interface {}",
            names.format_id((*entry_point).into()),
            names.format_id((*interface).into())
        ),
        (
            ValidationError::EntryPointInterfaceStorageClassDuplicate {
                entry_point,
                storage_class,
            },
            Some(names),
        ) => format!(
            "{} has duplicate {storage_class:?} interfaces",
            names.format_id((*entry_point).into())
        ),
        (
            ValidationError::EntryPointInterfaceLocationConflict {
                entry_point,
                storage_class,
                location,
                component,
                first_var,
                second_var,
            },
            Some(names),
        ) => format!(
            "{} has overlapping {storage_class:?} variables at location {location} component {component}: {} and {} both use this slot",
            names.format_id((*entry_point).into()),
            names.format_id((*first_var).into()),
            names.format_id((*second_var).into())
        ),
        (
            ValidationError::ExecutionModeRequiresExecutionModel {
                entry_point,
                mode,
                execution_model,
                allowed_models,
            },
            Some(names),
        ) => format!(
            "{} uses {execution_model:?} with mode {mode:?} (allowed: {:?})",
            names.format_id((*entry_point).into()),
            allowed_models
        ),
        (
            ValidationError::InvalidExecutionModeValue {
                entry_point,
                mode,
                value,
            },
            Some(names),
        ) => format!(
            "{} has {mode:?} value {value}",
            names.format_id((*entry_point).into())
        ),
        (
            ValidationError::EntryPointInterfaceFloatEncodingInvalid {
                interface,
                storage_class,
                encoding,
            },
            Some(names),
        ) => format!(
            "{} in {storage_class:?} uses {encoding:?}",
            names.format_id((*interface).into())
        ),
        (
            ValidationError::DuplicateEntryPoint {
                function,
                execution_model,
            },
            Some(names),
        ) => format!(
            "{} ({execution_model:?})",
            names.format_id((*function).into())
        ),
        (
            ValidationError::DuplicateEntryPointInterface {
                entry_point,
                interface,
            },
            Some(names),
        ) => format!(
            "{} duplicate interface {}",
            names.format_id((*entry_point).into()),
            names.format_id((*interface).into())
        ),
        (ValidationError::FunctionDeclarationAfterDefinition { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::MissingFunctionEntryBlock { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::MissingBlockTerminator { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (ValidationError::InstructionsAfterTerminator { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (ValidationError::MissingBlockTarget { function, target }, Some(names)) => format!(
            "{} missing block {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (
            ValidationError::FunctionCallTargetNotFunction {
                function,
                target,
                ..
            },
            Some(names),
        ) => format!(
            "{} calls non-function {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (
            ValidationError::MergeTargetMissing {
                function, target, ..
            },
            Some(names),
        ) => format!(
            "{} missing block {}",
            names.format_id((*function).into()),
            names.format_id((*target).into())
        ),
        (
            ValidationError::MergeInstructionNotBeforeTerminator {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::InvalidMergeTerminator {
                function, block, ..
            },
            Some(names),
        )
        | (ValidationError::DuplicateMergeInstruction { function, block }, Some(names))
        | (
            ValidationError::ContinueTargetMatchesMerge {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::MissingSelectionMerge {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::PhiIncomingTypeMismatch {
                function, block, ..
            },
            Some(names),
        )
        | (
            ValidationError::ValueDefinedInAnotherFunction {
                function,
                value: block,
            },
            Some(names),
        )
        | (
            ValidationError::FunctionVariableStorageClassMismatch {
                function,
                variable: block,
                ..
            },
            Some(names),
        )
        | (
            ValidationError::FunctionVariableNotInEntryBlock {
                function,
                variable: block,
            },
            Some(names),
        ) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (
            ValidationError::MissingBlockLabel {
                function,
                block_index,
            },
            Some(names),
        ) => format!(
            "{} missing OpLabel in block {}",
            names.format_id((*function).into()),
            block_index
        ),
        (
            ValidationError::MergeTargetIsBlock {
                function,
                block,
                kind,
                target,
            },
            Some(names),
        ) => format!(
            "{} uses {:?} target {} equal to its block {}",
            names.format_id((*function).into()),
            kind,
            names.format_id((*target).into()),
            names.format_id((*block).into()),
        ),
        (
            ValidationError::ValueNotDominated {
                function,
                block,
                value,
            },
            Some(names),
        ) => format!(
            "{} uses value {} in block {} before its definition dominates",
            names.format_id((*function).into()),
            names.format_id((*value).into()),
            names.format_id((*block).into()),
        ),
        (
            ValidationError::PhiIncomingNotDominated {
                function,
                block,
                incoming,
                value,
            },
            Some(names),
        ) => format!(
            "{} uses incoming value {} for predecessor {} in block {} before its definition dominates",
            names.format_id((*function).into()),
            names.format_id((*value).into()),
            names.format_id((*incoming).into()),
            names.format_id((*block).into()),
        ),
        (ValidationError::UndefinedId { function, id }, Some(names)) => {
            let func = function
                .map(|f| format!(" in function {}", names.format_id(f.into())))
                .unwrap_or_default();
            format!(
                "use of undefined id {}{}",
                names.format_id((*id).into()),
                func
            )
        }
        (
            ValidationError::ResultTypeNotType {
                instruction,
                result_type,
                found,
            },
            Some(names),
        ) => format!(
            "{:?} uses result type {} defined by non-type opcode {:?}",
            instruction,
            names.format_id((*result_type).into()),
            found
        ),
        (ValidationError::PhiAfterNonPhi { function, block }, Some(names)) => format!(
            "{} in block {}",
            names.format_id((*function).into()),
            names.format_id((*block).into())
        ),
        (ValidationError::MissingReturnValue { function, .. }, Some(names)) => {
            names.format_id((*function).into())
        }
        (ValidationError::ReturnValueInVoidFunction { function }, Some(names)) => {
            names.format_id((*function).into())
        }
        (
            ValidationError::FunctionTypeParameterVoid { type_id, parameter },
            Some(names),
        ) => {
            format!(
                "{} parameter {}",
                names.format_id((*type_id).into()),
                names.format_id((*parameter).into())
            )
        }
        (ValidationError::FunctionReturnTypeMismatch { function, .. }, Some(names))
        | (ValidationError::FunctionParameterCountMismatch { function, .. }, Some(names))
        | (ValidationError::FunctionParameterTypeMismatch { function, .. }, Some(names)) => {
            names.format_id((*function).into())
        }
        _ => error.to_string(),
    }
}

/// Attempts to render a validation error with friendly names derived from the provided module words.
pub fn format_validation_error_from_words(
    words: &[u32],
    options: &ValidationOptions,
    error: &ValidationError,
) -> String {
    if !options.use_friendly_names {
        return error.to_string();
    }
    let names = collect_friendly_names(words);
    format_validation_error(error, names.as_ref())
}

/// Builds a friendly-name table from a parsed module.
pub fn build_friendly_name_table(module: &Module) -> FriendlyNames {
    let mut id_names = HashMap::new();
    let mut member_names = HashMap::new();
    for inst in &module.debug_names {
        match inst.class.opcode {
            rspirv::spirv::Op::Name => {
                if let (
                    Some(rspirv::dr::Operand::IdRef(id)),
                    Some(rspirv::dr::Operand::LiteralString(name)),
                ) = (inst.operands.first(), inst.operands.get(1))
                {
                    id_names.insert(*id, name.clone());
                }
            }
            rspirv::spirv::Op::MemberName => {
                if let (
                    Some(rspirv::dr::Operand::IdRef(struct_id)),
                    Some(rspirv::dr::Operand::LiteralBit32(member)),
                    Some(rspirv::dr::Operand::LiteralString(name)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    member_names.insert((*struct_id, MemberIndex(*member)), name.clone());
                }
            }
            _ => {}
        }
    }
    FriendlyNames {
        id_names,
        member_names,
    }
}

fn collect_friendly_names(words: &[u32]) -> Option<FriendlyNames> {
    let mut loader = rspirv::dr::Loader::new();
    if rspirv::binary::parse_words(words, &mut loader).is_err() {
        return None;
    }
    let module = loader.module();
    Some(build_friendly_name_table(&module))
}
