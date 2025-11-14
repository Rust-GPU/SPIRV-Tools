/// Endianness of a SPIR-V module stream.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// Little-endian word ordering.
    Little = 0,
    /// Big-endian word ordering.
    Big = 1,
}

impl Endianness {
    /// Converts from the raw integer used in the C API.
    pub const fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Little),
            1 => Some(Self::Big),
            _ => None,
        }
    }

    /// Returns the raw representation expected by the C API.
    pub const fn to_raw(self) -> u32 {
        self as u32
    }
}

impl From<Endianness> for u32 {
    fn from(value: Endianness) -> Self {
        value.to_raw()
    }
}

impl TryFrom<u32> for Endianness {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::from_raw(value).ok_or(value)
    }
}

#[cfg(test)]
mod tests {
    use super::Endianness;

    #[test]
    fn round_trip() {
        for raw in 0..=1 {
            let value = Endianness::from_raw(raw).unwrap();
            assert_eq!(Endianness::from_raw(value.to_raw()), Some(value));
        }
        assert!(Endianness::from_raw(2).is_none());
    }
}
