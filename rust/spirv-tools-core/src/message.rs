/// Severity levels used by the message consumer callbacks.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MessageLevel {
    /// Unrecoverable environment error.
    Fatal = 0,
    /// Internal SPIRV-Tools failure.
    InternalError = 1,
    /// User input error.
    Error = 2,
    /// Warnings that do not abort execution.
    Warning = 3,
    /// Informational messages.
    Info = 4,
    /// Debug-only noise.
    Debug = 5,
}

impl MessageLevel {
    /// Converts a raw integer to a `MessageLevel`.
    pub const fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Fatal),
            1 => Some(Self::InternalError),
            2 => Some(Self::Error),
            3 => Some(Self::Warning),
            4 => Some(Self::Info),
            5 => Some(Self::Debug),
            _ => None,
        }
    }

    /// Returns the raw integer representation.
    pub const fn to_raw(self) -> u32 {
        self as u32
    }
}

impl From<MessageLevel> for u32 {
    fn from(level: MessageLevel) -> Self {
        level.to_raw()
    }
}

impl TryFrom<u32> for MessageLevel {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, <Self as TryFrom<u32>>::Error> {
        Self::from_raw(value).ok_or(value)
    }
}

#[cfg(test)]
mod tests {
    use super::MessageLevel;

    #[test]
    fn conversions_work() {
        for raw in 0..=5 {
            let level = MessageLevel::from_raw(raw).unwrap();
            assert_eq!(MessageLevel::from_raw(level.to_raw()), Some(level));
        }
        assert!(MessageLevel::from_raw(42).is_none());
    }
}
