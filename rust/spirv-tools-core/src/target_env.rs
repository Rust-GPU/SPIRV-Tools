/// Rust representation of `spv_target_env`.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TargetEnv {
    /// SPIR-V 1.0, no additional constraints.
    Universal1_0 = 0,
    /// Vulkan 1.0 semantics.
    Vulkan1_0 = 1,
    /// SPIR-V 1.1, no additional constraints.
    Universal1_1 = 2,
    /// OpenCL Full Profile 2.1 semantics.
    OpenCl2_1 = 3,
    /// OpenCL Full Profile 2.2 semantics.
    OpenCl2_2 = 4,
    /// OpenGL 4.0 semantics with GL_ARB_gl_spirv.
    OpenGl4_0 = 5,
    /// OpenGL 4.1 semantics with GL_ARB_gl_spirv.
    OpenGl4_1 = 6,
    /// OpenGL 4.2 semantics with GL_ARB_gl_spirv.
    OpenGl4_2 = 7,
    /// OpenGL 4.3 semantics with GL_ARB_gl_spirv.
    OpenGl4_3 = 8,
    /// OpenGL 4.5 semantics with GL_ARB_gl_spirv.
    OpenGl4_5 = 9,
    /// SPIR-V 1.2, no additional constraints.
    Universal1_2 = 10,
    /// OpenCL Full Profile 1.2 semantics.
    OpenCl1_2 = 11,
    /// OpenCL Embedded Profile 1.2 semantics.
    OpenClEmbedded1_2 = 12,
    /// OpenCL Full Profile 2.0 semantics.
    OpenCl2_0 = 13,
    /// OpenCL Embedded Profile 2.0 semantics.
    OpenClEmbedded2_0 = 14,
    /// OpenCL Embedded Profile 2.1 semantics.
    OpenClEmbedded2_1 = 15,
    /// OpenCL Embedded Profile 2.2 semantics.
    OpenClEmbedded2_2 = 16,
    /// SPIR-V 1.3, no additional constraints.
    Universal1_3 = 17,
    /// Vulkan 1.1 semantics.
    Vulkan1_1 = 18,
    /// Deprecated WebGPU environment.
    WebGpu0 = 19,
    /// SPIR-V 1.4, no additional constraints.
    Universal1_4 = 20,
    /// Vulkan 1.1 with VK_KHR_spirv_1_4 (SPIR-V 1.4 binary).
    Vulkan1_1Spirv1_4 = 21,
    /// SPIR-V 1.5, no additional constraints.
    Universal1_5 = 22,
    /// Vulkan 1.2 semantics.
    Vulkan1_2 = 23,
    /// SPIR-V 1.6, no additional constraints.
    Universal1_6 = 24,
    /// Vulkan 1.3 semantics.
    Vulkan1_3 = 25,
    /// Vulkan 1.4 semantics.
    Vulkan1_4 = 26,
    /// Sentinel used by the C API; not a valid environment.
    Max = 27,
}

impl TargetEnv {
    /// Converts from the raw integer used by the C API.
    pub const fn from_raw(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Universal1_0),
            1 => Some(Self::Vulkan1_0),
            2 => Some(Self::Universal1_1),
            3 => Some(Self::OpenCl2_1),
            4 => Some(Self::OpenCl2_2),
            5 => Some(Self::OpenGl4_0),
            6 => Some(Self::OpenGl4_1),
            7 => Some(Self::OpenGl4_2),
            8 => Some(Self::OpenGl4_3),
            9 => Some(Self::OpenGl4_5),
            10 => Some(Self::Universal1_2),
            11 => Some(Self::OpenCl1_2),
            12 => Some(Self::OpenClEmbedded1_2),
            13 => Some(Self::OpenCl2_0),
            14 => Some(Self::OpenClEmbedded2_0),
            15 => Some(Self::OpenClEmbedded2_1),
            16 => Some(Self::OpenClEmbedded2_2),
            17 => Some(Self::Universal1_3),
            18 => Some(Self::Vulkan1_1),
            19 => Some(Self::WebGpu0),
            20 => Some(Self::Universal1_4),
            21 => Some(Self::Vulkan1_1Spirv1_4),
            22 => Some(Self::Universal1_5),
            23 => Some(Self::Vulkan1_2),
            24 => Some(Self::Universal1_6),
            25 => Some(Self::Vulkan1_3),
            26 => Some(Self::Vulkan1_4),
            27 => Some(Self::Max),
            _ => None,
        }
    }

    /// Returns the raw integer representation.
    pub const fn to_raw(self) -> u32 {
        self as u32
    }

    /// Human-readable description mirroring `spvTargetEnvDescription`.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Universal1_0 => "SPIR-V 1.0",
            Self::Vulkan1_0 => "SPIR-V 1.0 (under Vulkan 1.0 semantics)",
            Self::Universal1_1 => "SPIR-V 1.1",
            Self::OpenCl2_1 => "SPIR-V 1.0 (under OpenCL 2.1 Full Profile semantics)",
            Self::OpenCl2_2 => "SPIR-V 1.2 (under OpenCL 2.2 Full Profile semantics)",
            Self::OpenGl4_0 => "SPIR-V 1.0 (under OpenGL 4.0 semantics)",
            Self::OpenGl4_1 => "SPIR-V 1.0 (under OpenGL 4.1 semantics)",
            Self::OpenGl4_2 => "SPIR-V 1.0 (under OpenGL 4.2 semantics)",
            Self::OpenGl4_3 => "SPIR-V 1.0 (under OpenGL 4.3 semantics)",
            Self::OpenGl4_5 => "SPIR-V 1.0 (under OpenGL 4.5 semantics)",
            Self::Universal1_2 => "SPIR-V 1.2",
            Self::OpenCl1_2 => "SPIR-V 1.0 (under OpenCL 1.2 Full Profile semantics)",
            Self::OpenClEmbedded1_2 => "SPIR-V 1.0 (under OpenCL 1.2 Embedded Profile semantics)",
            Self::OpenCl2_0 => "SPIR-V 1.0 (under OpenCL 2.0 Full Profile semantics)",
            Self::OpenClEmbedded2_0 => "SPIR-V 1.0 (under OpenCL 2.0 Embedded Profile semantics)",
            Self::OpenClEmbedded2_1 => "SPIR-V 1.0 (under OpenCL 2.1 Embedded Profile semantics)",
            Self::OpenClEmbedded2_2 => "SPIR-V 1.2 (under OpenCL 2.2 Embedded Profile semantics)",
            Self::Universal1_3 => "SPIR-V 1.3",
            Self::Vulkan1_1 => "SPIR-V 1.3 (under Vulkan 1.1 semantics)",
            Self::WebGpu0 => "",
            Self::Universal1_4 => "SPIR-V 1.4",
            Self::Vulkan1_1Spirv1_4 => "SPIR-V 1.4 (under Vulkan 1.1 semantics)",
            Self::Universal1_5 => "SPIR-V 1.5",
            Self::Vulkan1_2 => "SPIR-V 1.5 (under Vulkan 1.2 semantics)",
            Self::Universal1_6 => "SPIR-V 1.6",
            Self::Vulkan1_3 => "SPIR-V 1.6 (under Vulkan 1.3 semantics)",
            Self::Vulkan1_4 => "SPIR-V 1.6 (under Vulkan 1.4 semantics)",
            Self::Max => "",
        }
    }
}

impl From<TargetEnv> for u32 {
    fn from(value: TargetEnv) -> Self {
        value.to_raw()
    }
}

impl TryFrom<u32> for TargetEnv {
    type Error = u32;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::from_raw(value).ok_or(value)
    }
}

#[cfg(test)]
mod tests {
    use super::TargetEnv;

    #[test]
    fn round_trip() {
        for raw in 0..=27 {
            let env = TargetEnv::from_raw(raw).unwrap();
            assert_eq!(TargetEnv::from_raw(env.to_raw()), Some(env));
        }
        assert!(TargetEnv::from_raw(100).is_none());
    }

    #[test]
    fn descriptions_match_known_values() {
        assert_eq!(
            TargetEnv::Vulkan1_0.description(),
            "SPIR-V 1.0 (under Vulkan 1.0 semantics)"
        );
        assert_eq!(TargetEnv::Universal1_5.description(), "SPIR-V 1.5");
    }
}
