use std::fmt;

use thiserror::Error;

/// Typed representation of `spv_result_t` with exhaustive variants.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SpvResult {
    /// Operation completed successfully.
    Success = 0,
    /// Operation is unsupported in the current configuration.
    Unsupported = 1,
    /// Input stream ended before completion.
    EndOfStream = 2,
    /// Operation succeeded with a warning.
    Warning = 3,
    /// Failed to match expected pattern.
    FailedMatch = 4,
    /// Operation succeeded but requested early termination.
    RequestedTermination = 5,
    /// Internal error inside SPIRV-Tools.
    ErrorInternal = -1,
    /// Host system ran out of memory.
    ErrorOutOfMemory = -2,
    /// Encountered invalid pointer input.
    ErrorInvalidPointer = -3,
    /// Encountered malformed binary input.
    ErrorInvalidBinary = -4,
    /// Encountered malformed text input.
    ErrorInvalidText = -5,
    /// Lookup table was invalid or corrupt.
    ErrorInvalidTable = -6,
    /// Encountered invalid value.
    ErrorInvalidValue = -7,
    /// Diagnostic object was invalid.
    ErrorInvalidDiagnostic = -8,
    /// Table lookup failed because the key was invalid.
    ErrorInvalidLookup = -9,
    /// Referenced ID was invalid.
    ErrorInvalidId = -10,
    /// Control-flow graph was invalid.
    ErrorInvalidCfg = -11,
    /// Module layout was invalid.
    ErrorInvalidLayout = -12,
    /// Capability requirements were invalid.
    ErrorInvalidCapability = -13,
    /// Module data failed validation.
    ErrorInvalidData = -14,
    /// Required extension is missing.
    ErrorMissingExtension = -15,
    /// Wrong SPIR-V version.
    ErrorWrongVersion = -16,
    /// Error related to `SPV_INTEL_function_variants`.
    ErrorFunctionVariant = -17,
}

impl SpvResult {
    /// Converts the raw `i32` representation into `SpvResult`.
    pub const fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Success),
            1 => Some(Self::Unsupported),
            2 => Some(Self::EndOfStream),
            3 => Some(Self::Warning),
            4 => Some(Self::FailedMatch),
            5 => Some(Self::RequestedTermination),
            -1 => Some(Self::ErrorInternal),
            -2 => Some(Self::ErrorOutOfMemory),
            -3 => Some(Self::ErrorInvalidPointer),
            -4 => Some(Self::ErrorInvalidBinary),
            -5 => Some(Self::ErrorInvalidText),
            -6 => Some(Self::ErrorInvalidTable),
            -7 => Some(Self::ErrorInvalidValue),
            -8 => Some(Self::ErrorInvalidDiagnostic),
            -9 => Some(Self::ErrorInvalidLookup),
            -10 => Some(Self::ErrorInvalidId),
            -11 => Some(Self::ErrorInvalidCfg),
            -12 => Some(Self::ErrorInvalidLayout),
            -13 => Some(Self::ErrorInvalidCapability),
            -14 => Some(Self::ErrorInvalidData),
            -15 => Some(Self::ErrorMissingExtension),
            -16 => Some(Self::ErrorWrongVersion),
            -17 => Some(Self::ErrorFunctionVariant),
            _ => None,
        }
    }

    /// Returns the raw `i32` representation expected by the C API.
    pub const fn to_raw(self) -> i32 {
        self as i32
    }

    /// Returns `true` if this result represents a successful outcome.
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success | Self::RequestedTermination)
    }

    /// Converts the value into a standard `Result`.
    pub const fn into_result(self) -> Result<(), Self> {
        if self.is_success() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

impl TryFrom<i32> for SpvResult {
    type Error = InvalidSpvResult;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Self::from_raw(value).ok_or(InvalidSpvResult(value))
    }
}

impl From<SpvResult> for i32 {
    fn from(value: SpvResult) -> Self {
        value.to_raw()
    }
}

/// Error returned when an unknown `spv_result_t` discriminant is encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("unknown spv_result_t value {0}")]
pub struct InvalidSpvResult(pub i32);

impl fmt::Display for SpvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::SpvResult;

    #[test]
    fn round_trip_success_values() {
        for value in -17..=5 {
            if let Some(result) = SpvResult::from_raw(value) {
                assert_eq!(SpvResult::from_raw(result.to_raw()), Some(result));
            }
        }
    }

    #[test]
    fn failure_detection() {
        assert!(SpvResult::Success.is_success());
        assert!(SpvResult::RequestedTermination.is_success());
        assert!(!SpvResult::ErrorInvalidId.is_success());
    }
}
