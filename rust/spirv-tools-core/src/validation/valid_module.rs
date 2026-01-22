//! Validated module and caching.
//!
//! This module provides:
//! - `ValidModule`: A successfully validated SPIR-V module
//! - `ValidModuleCache`: A cache for validated modules
//! - `MaybeValidModule`: Input that can be validated (binary or text)
//! - `ValidatableModule`: Trait for types that can be validated

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use rspirv::dr::Module;

use crate::target_env::TargetEnv;
use crate::version::SpirvVersion;

use super::error::ValidationError;
use super::friendly_names::FriendlyNames;
use super::header::ValidatedHeader;
use super::options::ValidationOptions;
use super::span::SpannedValidationError;
use super::types::ModuleWords;

/// A validated module containing the original binary plus the parsed representation.
#[derive(Debug)]
pub struct ValidModule {
    pub(crate) words: ModuleWords,
    pub(crate) module: Module,
    pub(crate) env: TargetEnv,
    pub(crate) header: ValidatedHeader,
    pub(crate) effective_version: SpirvVersion,
    pub(crate) options: ValidationOptions,
    pub(crate) friendly_names: Option<FriendlyNames>,
}

impl ValidModule {
    /// Returns the validated words that were successfully checked.
    pub fn words(&self) -> &[u32] {
        self.words.as_slice()
    }

    /// Returns the parsed module corresponding to the validated words.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Returns the target environment this module was validated against.
    pub fn env(&self) -> TargetEnv {
        self.env
    }

    /// Returns the SPIR-V version actually used during validation (module version clamped to env).
    pub fn effective_version(&self) -> SpirvVersion {
        self.effective_version
    }

    /// Returns the declared SPIR-V version from the module header.
    pub fn module_version(&self) -> SpirvVersion {
        self.header.version()
    }

    /// Returns the validated module header.
    pub fn header(&self) -> ValidatedHeader {
        self.header
    }

    /// Returns a shared handle to the validated words.
    pub fn words_handle(&self) -> ModuleWords {
        self.words.clone()
    }

    /// Returns the validator options applied during validation.
    pub fn options(&self) -> &ValidationOptions {
        &self.options
    }

    /// Returns friendly names applied during validation (if enabled).
    pub fn friendly_names(&self) -> Option<&FriendlyNames> {
        self.friendly_names.as_ref()
    }
}

/// A cache of validated modules keyed by target environment and module contents.
#[derive(Default)]
pub struct ValidModuleCache {
    entries: HashMap<(TargetEnv, u64, ValidationOptions), Arc<ValidModule>>,
}

impl ValidModuleCache {
    /// Validate the provided binary words, returning a shared validated module and caching the result.
    pub fn validate_words(
        &mut self,
        words: &[u32],
        env: TargetEnv,
    ) -> Result<Arc<ValidModule>, ValidationError> {
        self.validate_words_with_options(words, env, ValidationOptions::default())
    }

    /// Validate with explicit options.
    pub fn validate_words_with_options(
        &mut self,
        words: &[u32],
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<Arc<ValidModule>, ValidationError> {
        let hash = hash_words(words, env);
        if let Some(cached) = self.entries.get(&(env, hash, options.clone())) {
            if cached.words_handle().as_slice() == words {
                return Ok(Arc::clone(cached));
            }
        }
        let validated = super::validate_words_internal(
            ModuleWords::from(Arc::from(words)),
            env,
            options.clone(),
            None,
        )
        .map_err(|e: SpannedValidationError| e.error)?;
        let validated = Arc::new(validated);
        self.entries
            .insert((env, hash, options), Arc::clone(&validated));
        Ok(validated)
    }
}

pub(crate) fn hash_words(words: &[u32], env: TargetEnv) -> u64 {
    let mut hasher = DefaultHasher::new();
    env.hash(&mut hasher);
    words.len().hash(&mut hasher);
    for word in words {
        word.hash(&mut hasher);
    }
    hasher.finish()
}

/// Input sources that can be validated before becoming a `ValidModule`.
pub enum MaybeValidModule<'a> {
    /// Pre-assembled SPIR-V words.
    Binary(&'a [u32]),
    /// SPIR-V assembly text to be assembled and validated.
    Text(&'a str),
}

impl<'a> MaybeValidModule<'a> {
    /// Validate the provided input, assembling text when necessary.
    pub fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError> {
        self.validate_with_options(env, ValidationOptions::default())
    }

    /// Validate the provided input with explicit options, assembling text when necessary.
    pub fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        match self {
            MaybeValidModule::Binary(words) => super::validate_words_internal(
                ModuleWords::from(Arc::from(words)),
                env,
                options,
                None,
            )
            .map_err(|e: SpannedValidationError| e.error),
            MaybeValidModule::Text(text) => {
                let binary = ModuleWords::from(Arc::<[u32]>::from(
                    crate::assembly::assemble_text(text)
                        .map_err(|err| ValidationError::Parse(err.to_string()))?
                        .into_boxed_slice(),
                ));
                super::validate_words_internal(binary, env, options, None)
                    .map_err(|e: SpannedValidationError| e.error)
            }
        }
    }
}

/// Convenience trait for validating either binary words or assembly text.
pub trait ValidatableModule<'a> {
    /// Validates the module input for the requested target environment.
    fn validate(self, env: TargetEnv) -> Result<ValidModule, ValidationError>
    where
        Self: Sized,
    {
        self.validate_with_options(env, ValidationOptions::default())
    }

    /// Validates the module input for the requested target environment with explicit options.
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError>
    where
        Self: Sized;
}

impl<'a> ValidatableModule<'a> for &'a [u32] {
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Binary(self).validate_with_options(env, options)
    }
}

impl<'a> ValidatableModule<'a> for &'a str {
    fn validate_with_options(
        self,
        env: TargetEnv,
        options: ValidationOptions,
    ) -> Result<ValidModule, ValidationError> {
        MaybeValidModule::Text(self).validate_with_options(env, options)
    }
}
