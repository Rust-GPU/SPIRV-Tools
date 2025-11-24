/// Encodes a SPIR-V version as used in binary module headers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SpirvVersion {
    major: u8,
    minor: u8,
}

impl SpirvVersion {
    /// Constructs a new SPIR-V version.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Returns the major component of the version.
    pub const fn major(self) -> u8 {
        self.major
    }

    /// Returns the minor component of the version.
    pub const fn minor(self) -> u8 {
        self.minor
    }

    /// Parses a packed SPIR-V version word.
    pub const fn from_word(word: u32) -> Self {
        Self {
            major: ((word >> 16) & 0xff) as u8,
            minor: ((word >> 8) & 0xff) as u8,
        }
    }

    /// Returns the packed 32-bit representation defined by the SPIR-V spec.
    pub const fn to_word(self) -> u32 {
        ((self.major as u32) << 16) | ((self.minor as u32) << 8)
    }

    /// Returns `true` if `self` covers at least the requested version.
    pub const fn meets_or_exceeds(self, other: Self) -> bool {
        let self_word = self.to_word();
        let other_word = other.to_word();
        self_word >= other_word
    }
}

/// Encodes a Vulkan API version using the packed bit layout defined by Vulkan.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VulkanVersion(u32);

impl VulkanVersion {
    /// Creates a version wrapper from the packed 32-bit representation.
    pub const fn from_word(word: u32) -> Self {
        Self(word)
    }

    /// Returns the packed representation.
    pub const fn to_word(self) -> u32 {
        self.0
    }

    /// Returns `true` if `self` satisfies (>=) the requested version.
    pub const fn meets_or_exceeds(self, other: Self) -> bool {
        self.0 >= other.0
    }
}

#[cfg(test)]
mod tests {
    use super::{SpirvVersion, VulkanVersion};

    #[test]
    fn spirv_to_word_matches_spec() {
        let v = SpirvVersion::new(1, 5);
        assert_eq!(v.to_word(), 0x10500);
        let parsed = SpirvVersion::from_word(0x10500);
        assert_eq!(parsed, v);
    }

    #[test]
    fn vulkan_ordering() {
        let a = VulkanVersion::from_word(1u32 << 22);
        let b = VulkanVersion::from_word((1u32 << 22) | (1u32 << 12));
        assert!(b.meets_or_exceeds(a));
        assert!(!a.meets_or_exceeds(b));
    }
}
