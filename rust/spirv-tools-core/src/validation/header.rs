//! Validated module header.
//!
//! This module provides the `ValidatedHeader` type which represents a SPIR-V
//! module header that has passed basic validation checks (schema, version, bound).

use rspirv::dr::Module;

use crate::version::SpirvVersion;

use super::error::ValidationError;
use super::types::{CheckedBound, DeclaredBound, Schema};

/// A validated module header with a checked bound and schema.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ValidatedHeader {
    version: SpirvVersion,
    bound: CheckedBound,
    schema: Schema,
}

impl ValidatedHeader {
    /// Creates a validated header from its components.
    pub fn new(version: SpirvVersion, bound: CheckedBound, schema: Schema) -> Self {
        Self {
            version,
            bound,
            schema,
        }
    }

    /// Parses and validates a module header, ensuring the bound and schema are valid.
    pub fn from_module(module: &Module) -> Result<Self, ValidationError> {
        let header = module
            .header
            .as_ref()
            .ok_or(ValidationError::MissingHeader)?;
        let schema = Schema::validate(header.reserved_word)?;
        let version = SpirvVersion::from_word(header.version);
        let declared_bound = DeclaredBound(header.bound);
        let bound = CheckedBound::new(declared_bound).ok_or(ValidationError::InvalidIdBound {
            bound: declared_bound,
        })?;
        Ok(Self {
            version,
            bound,
            schema,
        })
    }

    /// Returns the validated id bound associated with this header.
    pub fn bound(self) -> CheckedBound {
        self.bound
    }

    /// Returns the module's declared SPIR-V version.
    pub fn version(self) -> SpirvVersion {
        self.version
    }

    /// Returns the validated schema value (always zero for valid modules).
    pub fn schema(self) -> Schema {
        self.schema
    }
}
