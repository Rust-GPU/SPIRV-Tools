//! Binary manipulation utilities for SPIR-V.

use crate::error::{Error, SpirvResult};
use std::fmt;

/// A SPIR-V binary, which can be either externally owned or owned by this struct.
pub enum Binary {
    OwnedU32(Vec<u32>),
    OwnedU8(Vec<u8>),
}

impl Binary {
    /// Gets a byte array for binary.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }

    /// Gets the words for the binary.
    #[inline]
    pub fn as_words(&self) -> &[u32] {
        self.as_ref()
    }
}

impl TryFrom<Vec<u8>> for Binary {
    type Error = Error;

    #[inline]
    fn try_from(v: Vec<u8>) -> Result<Self, Self::Error> {
        if !v.len().is_multiple_of(size_of::<u32>()) {
            Err(Error {
                inner: SpirvResult::InvalidBinary,
                diagnostic: None,
            })
        } else {
            Ok(Binary::OwnedU8(v))
        }
    }
}

impl AsRef<[u32]> for Binary {
    #[inline]
    fn as_ref(&self) -> &[u32] {
        match self {
            Self::OwnedU32(v) => v,
            Self::OwnedU8(v) => {
                // If you hit a panic here it's because try_from wasn't used
                to_binary(v).unwrap()
            }
        }
    }
}

impl AsRef<[u8]> for Binary {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::OwnedU32(v) => from_binary(v),
            Self::OwnedU8(v) => v,
        }
    }
}

impl fmt::Debug for Binary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = match self {
            Self::OwnedU32(_) => f.debug_struct("OwnedU32"),
            Self::OwnedU8(_) => f.debug_struct("OwnedU8"),
        };

        ds.field("word_count", &self.as_words().len()).finish()
    }
}

/// Transmutes a SPIR-V binary, which are stored as 32 bit words, into a more
/// digestible byte array.
#[inline]
pub fn from_binary(bin: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(bin.as_ptr().cast(), size_of_val(bin)) }
}

/// Transmutes a regular byte array into a SPIR-V binary of 32 bit words.
/// Fails if the input length is not a multiple of 4 or not properly aligned.
#[inline]
pub fn to_binary(bytes: &[u8]) -> Result<&[u32], Error> {
    if !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(Error {
            inner: SpirvResult::InvalidBinary,
            diagnostic: None,
        });
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(size_of::<u32>()) {
        return Err(Error {
            inner: SpirvResult::InvalidBinary,
            diagnostic: None,
        });
    }

    Ok(
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size_of::<u32>())
        },
    )
}
