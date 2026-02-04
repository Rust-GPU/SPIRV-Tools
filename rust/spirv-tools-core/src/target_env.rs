use crate::{
    validation::ExtensionName,
    version::{SpirvVersion, VulkanVersion},
};
mod extension_allowlist;
use extension_allowlist::{ExtensionAllowlist, EXTENSION_ALLOWLIST};

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
    const ORDERED_UNIVERSAL_ENVS: [TargetEnv; 7] = [
        TargetEnv::Universal1_0,
        TargetEnv::Universal1_1,
        TargetEnv::Universal1_2,
        TargetEnv::Universal1_3,
        TargetEnv::Universal1_4,
        TargetEnv::Universal1_5,
        TargetEnv::Universal1_6,
    ];

    const fn vulkan_word(major: u32, minor: u32) -> u32 {
        (major << 22) | (minor << 12)
    }

    const TARGET_NAMES: &'static [(&'static str, TargetEnv)] = &[
        ("vulkan1.0", TargetEnv::Vulkan1_0),
        ("vulkan1.1spv1.4", TargetEnv::Vulkan1_1Spirv1_4),
        ("vulkan1.1", TargetEnv::Vulkan1_1),
        ("vulkan1.2", TargetEnv::Vulkan1_2),
        ("vulkan1.3", TargetEnv::Vulkan1_3),
        ("vulkan1.4", TargetEnv::Vulkan1_4),
        ("spv1.0", TargetEnv::Universal1_0),
        ("spv1.1", TargetEnv::Universal1_1),
        ("spv1.2", TargetEnv::Universal1_2),
        ("spv1.3", TargetEnv::Universal1_3),
        ("spv1.4", TargetEnv::Universal1_4),
        ("spv1.5", TargetEnv::Universal1_5),
        ("spv1.6", TargetEnv::Universal1_6),
        ("opencl1.2embedded", TargetEnv::OpenClEmbedded1_2),
        ("opencl1.2", TargetEnv::OpenCl1_2),
        ("opencl2.0embedded", TargetEnv::OpenClEmbedded2_0),
        ("opencl2.0", TargetEnv::OpenCl2_0),
        ("opencl2.1embedded", TargetEnv::OpenClEmbedded2_1),
        ("opencl2.1", TargetEnv::OpenCl2_1),
        ("opencl2.2embedded", TargetEnv::OpenClEmbedded2_2),
        ("opencl2.2", TargetEnv::OpenCl2_2),
        ("opengl4.0", TargetEnv::OpenGl4_0),
        ("opengl4.1", TargetEnv::OpenGl4_1),
        ("opengl4.2", TargetEnv::OpenGl4_2),
        ("opengl4.3", TargetEnv::OpenGl4_3),
        ("opengl4.5", TargetEnv::OpenGl4_5),
    ];

    const VULKAN_ENV_TABLE: &'static [(TargetEnv, VulkanVersion, SpirvVersion)] = &[
        (
            TargetEnv::Vulkan1_0,
            VulkanVersion::from_word(Self::vulkan_word(1, 0)),
            SpirvVersion::new(1, 0),
        ),
        (
            TargetEnv::Vulkan1_1,
            VulkanVersion::from_word(Self::vulkan_word(1, 1)),
            SpirvVersion::new(1, 3),
        ),
        (
            TargetEnv::Vulkan1_1Spirv1_4,
            VulkanVersion::from_word(Self::vulkan_word(1, 1)),
            SpirvVersion::new(1, 4),
        ),
        (
            TargetEnv::Vulkan1_2,
            VulkanVersion::from_word(Self::vulkan_word(1, 2)),
            SpirvVersion::new(1, 5),
        ),
        (
            TargetEnv::Vulkan1_3,
            VulkanVersion::from_word(Self::vulkan_word(1, 3)),
            SpirvVersion::new(1, 6),
        ),
        (
            TargetEnv::Vulkan1_4,
            VulkanVersion::from_word(Self::vulkan_word(1, 4)),
            SpirvVersion::new(1, 6),
        ),
    ];

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

    /// Returns the associated SPIR-V version for the target environment.
    pub const fn spirv_version(self) -> SpirvVersion {
        match self {
            Self::Universal1_0
            | Self::Vulkan1_0
            | Self::OpenCl1_2
            | Self::OpenClEmbedded1_2
            | Self::OpenCl2_0
            | Self::OpenClEmbedded2_0
            | Self::OpenCl2_1
            | Self::OpenClEmbedded2_1
            | Self::OpenGl4_0
            | Self::OpenGl4_1
            | Self::OpenGl4_2
            | Self::OpenGl4_3
            | Self::OpenGl4_5 => SpirvVersion::new(1, 0),
            Self::Universal1_1 => SpirvVersion::new(1, 1),
            Self::Universal1_2 | Self::OpenCl2_2 | Self::OpenClEmbedded2_2 => {
                SpirvVersion::new(1, 2)
            }
            Self::Universal1_3 | Self::Vulkan1_1 => SpirvVersion::new(1, 3),
            Self::Universal1_4 | Self::Vulkan1_1Spirv1_4 => SpirvVersion::new(1, 4),
            Self::Universal1_5 | Self::Vulkan1_2 => SpirvVersion::new(1, 5),
            Self::Universal1_6 | Self::Vulkan1_3 | Self::Vulkan1_4 => SpirvVersion::new(1, 6),
            Self::WebGpu0 | Self::Max => SpirvVersion::new(0, 0),
        }
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

    /// Returns true if the environment belongs to the Vulkan profile family.
    pub const fn is_vulkan(self) -> bool {
        matches!(
            self,
            Self::Vulkan1_0
                | Self::Vulkan1_1
                | Self::Vulkan1_1Spirv1_4
                | Self::Vulkan1_2
                | Self::Vulkan1_3
                | Self::Vulkan1_4
        )
    }

    /// Returns true if the environment belongs to the Universal profile family.
    pub const fn is_universal(self) -> bool {
        matches!(
            self,
            Self::Universal1_0
                | Self::Universal1_1
                | Self::Universal1_2
                | Self::Universal1_3
                | Self::Universal1_4
                | Self::Universal1_5
                | Self::Universal1_6
        )
    }

    /// Returns true if the environment belongs to the OpenCL profile family.
    pub const fn is_opencl(self) -> bool {
        matches!(
            self,
            Self::OpenCl1_2
                | Self::OpenClEmbedded1_2
                | Self::OpenCl2_0
                | Self::OpenClEmbedded2_0
                | Self::OpenClEmbedded2_1
                | Self::OpenClEmbedded2_2
                | Self::OpenCl2_1
                | Self::OpenCl2_2
        )
    }

    /// Returns true if the environment is specifically OpenCL 1.2 (full or embedded).
    ///
    /// This is used for validation rules that differ between OpenCL 1.2 and later versions,
    /// such as the prohibition of Generic storage class for atomics in OpenCL 1.2.
    pub const fn is_opencl_1_2(self) -> bool {
        matches!(self, Self::OpenCl1_2 | Self::OpenClEmbedded1_2)
    }

    /// Returns true if the environment belongs to the OpenGL profile family.
    pub const fn is_opengl(self) -> bool {
        matches!(
            self,
            Self::OpenGl4_0 | Self::OpenGl4_1 | Self::OpenGl4_2 | Self::OpenGl4_3 | Self::OpenGl4_5
        )
    }

    /// Returns true if the environment is a well-defined, non-deprecated value.
    pub const fn is_valid(self) -> bool {
        !matches!(self, Self::WebGpu0 | Self::Max)
    }

    /// Returns the log namespace used by diagnostics.
    pub const fn log_namespace(self) -> &'static str {
        match self {
            Self::OpenCl1_2
            | Self::OpenCl2_0
            | Self::OpenCl2_1
            | Self::OpenCl2_2
            | Self::OpenClEmbedded1_2
            | Self::OpenClEmbedded2_0
            | Self::OpenClEmbedded2_1
            | Self::OpenClEmbedded2_2 => "OpenCL",
            Self::OpenGl4_0
            | Self::OpenGl4_1
            | Self::OpenGl4_2
            | Self::OpenGl4_3
            | Self::OpenGl4_5 => "OpenGL",
            Self::Vulkan1_0
            | Self::Vulkan1_1
            | Self::Vulkan1_1Spirv1_4
            | Self::Vulkan1_2
            | Self::Vulkan1_3
            | Self::Vulkan1_4 => "Vulkan",
            Self::Universal1_0
            | Self::Universal1_1
            | Self::Universal1_2
            | Self::Universal1_3
            | Self::Universal1_4
            | Self::Universal1_5
            | Self::Universal1_6 => "Universal",
            Self::WebGpu0 | Self::Max => "Unknown",
        }
    }

    /// Attempts to parse a textual prefix (e.g. "vulkan1.2") into an env.
    pub fn parse_name(input: &str) -> Option<Self> {
        for (name, env) in Self::TARGET_NAMES {
            if input.starts_with(name) {
                return Some(*env);
            }
        }
        None
    }

    /// Attempts to resolve the least-capable Vulkan environment satisfying
    /// the requested Vulkan/SPIR-V versions.
    pub fn parse_vulkan_env(vulkan: u32, spirv: u32) -> Option<Self> {
        let vulkan = VulkanVersion::from_word(vulkan);
        let spirv = SpirvVersion::from_word(spirv);
        for &(env, min_vulkan, max_spirv) in Self::VULKAN_ENV_TABLE {
            if min_vulkan.meets_or_exceeds(vulkan) && max_spirv.meets_or_exceeds(spirv) {
                return Some(env);
            }
        }
        None
    }

    /// Returns a textual list of the environment prefixes, formatted for CLI help.
    pub fn list_target_envs(pad: usize, wrap: usize) -> String {
        let mut result = String::new();
        let mut line = String::new();
        let mut max_line_len = wrap.saturating_sub(pad);
        let mut first_on_line = true;

        for (name, _) in Self::TARGET_NAMES {
            let word_len = name.len() + if first_on_line { 0 } else { 1 };
            if !line.is_empty() && max_line_len > 0 && line.len() + word_len > max_line_len {
                result.push_str(&line);
                result.push('\n');
                line.clear();
                if pad > 0 {
                    line.push_str(&" ".repeat(pad));
                }
                max_line_len = wrap;
                first_on_line = true;
            }

            if !first_on_line {
                line.push('|');
            }
            line.push_str(name);
            first_on_line = false;
        }

        result.push_str(&line);
        result
    }
}

/// Parses the textual header emitted by `spirv-dis` to determine the target env.
pub fn read_env_from_text(text: &[u8]) -> Option<TargetEnv> {
    const PREFIX: &[u8] = b"; Version: 1.";
    let mut i = 0usize;
    while i < text.len() {
        let byte = text[i];
        if byte == b';' {
            if i + PREFIX.len() >= text.len() {
                return None;
            }

            if text[i..].starts_with(PREFIX) {
                let minor_pos = i + PREFIX.len();
                let minor = text[minor_pos];
                let next = text.get(minor_pos + 1).copied();
                if minor.is_ascii_digit() && !next.map(|b| b.is_ascii_digit()).unwrap_or(false) {
                    let index = (minor - b'0') as usize;
                    if let Some(&env) = TargetEnv::ORDERED_UNIVERSAL_ENVS.get(index) {
                        return Some(env);
                    }
                }
            }

            while i < text.len() && text[i] != b'\n' {
                i += 1;
            }
        } else if !byte.is_ascii_whitespace() {
            break;
        }
        i += 1;
    }
    None
}

impl TargetEnv {
    /// Returns whether an extension is permitted for this target environment.
    ///
    /// The WebGPU environment forbids all extensions. Universal environments allow
    /// all extensions (matching the C++ SPIRV-Tools behavior where Universal targets
    /// have no extension restrictions). Other environments consult the generated
    /// allowlist derived from the extension registry.
    pub fn is_extension_allowed(self, extension: &ExtensionName) -> bool {
        if matches!(self, TargetEnv::WebGpu0) {
            return false;
        }
        // Universal environments allow all extensions, matching the C++ SPIRV-Tools
        // behavior where the capability/extension validation pass simply returns
        // SPV_SUCCESS for Universal targets without checking.
        if self.is_universal() {
            return true;
        }
        let name = extension.as_str();
        let normalized = name.to_ascii_lowercase();
        if normalized.contains("opencl") {
            return self.is_opencl();
        }
        let allowlist = EXTENSION_ALLOWLIST
            .iter()
            .find(|(known, _)| known.eq_ignore_ascii_case(name))
            .map(|(_, rule)| rule)
            .copied()
            .unwrap_or(ExtensionAllowlist {
                allow_vulkan: true,
                allow_opencl: true,
                allow_opengl: true,
                allow_universal: true,
            });
        allowlist.allowed_for(self)
    }

    /// Returns whether a capability is permitted for this target environment.
    ///
    /// The WebGPU environment allows only the core Shader capability. Vulkan and OpenCL
    /// allowlists follow the C++ validator tables so optional capabilities are permitted
    /// when the environment version supports them. Other environments fall back to the
    /// default (permissive) rules from the legacy validator.
    pub fn is_capability_allowed(self, capability: rspirv::spirv::Capability) -> bool {
        use rspirv::spirv::Capability;
        match self {
            TargetEnv::WebGpu0 => capability == Capability::Shader,
            TargetEnv::Vulkan1_0 => {
                is_support_guaranteed_vulkan_1_0(capability)
                    || is_support_optional_vulkan_1_0(capability)
            }
            TargetEnv::Vulkan1_1 | TargetEnv::Vulkan1_1Spirv1_4 => {
                is_support_guaranteed_vulkan_1_1(capability)
                    || is_support_optional_vulkan_1_1(capability)
            }
            TargetEnv::Vulkan1_2 => {
                is_support_guaranteed_vulkan_1_2(capability)
                    || is_support_optional_vulkan_1_2(capability)
            }
            TargetEnv::Vulkan1_3 => {
                is_support_guaranteed_vulkan_1_3(capability)
                    || is_support_optional_vulkan_1_3(capability)
            }
            TargetEnv::Vulkan1_4 => {
                is_support_guaranteed_vulkan_1_4(capability)
                    || is_support_optional_vulkan_1_4(capability)
            }
            env if env.is_opencl() => is_opencl_capability_allowed(env, capability),
            _ => true,
        }
    }
}

fn is_support_guaranteed_vulkan_1_0(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    matches!(
        capability,
        Matrix
            | Shader
            | InputAttachment
            | Sampled1D
            | Image1D
            | SampledBuffer
            | ImageBuffer
            | ImageQuery
            | DerivativeControl
    )
}

fn is_support_guaranteed_vulkan_1_1(capability: rspirv::spirv::Capability) -> bool {
    is_support_guaranteed_vulkan_1_0(capability)
        || matches!(
            capability,
            rspirv::spirv::Capability::DeviceGroup | rspirv::spirv::Capability::MultiView
        )
}

fn is_support_guaranteed_vulkan_1_2(capability: rspirv::spirv::Capability) -> bool {
    is_support_guaranteed_vulkan_1_1(capability)
        || matches!(capability, rspirv::spirv::Capability::ShaderNonUniform)
}

fn is_support_guaranteed_vulkan_1_3(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_guaranteed_vulkan_1_2(capability)
        || matches!(
            capability,
            DotProduct
                | DotProductInputAll
                | DotProductInput4x8Bit
                | DotProductInput4x8BitPacked
                | VulkanMemoryModel
                | VulkanMemoryModelDeviceScope
                | PhysicalStorageBufferAddresses
                | DemoteToHelperInvocation
        )
}

fn is_support_guaranteed_vulkan_1_4(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_guaranteed_vulkan_1_3(capability)
        || matches!(
            capability,
            UniformBufferArrayDynamicIndexing
                | SampledImageArrayDynamicIndexing
                | StorageBufferArrayDynamicIndexing
                | StorageImageArrayDynamicIndexing
                | Int16
                | StorageBuffer16BitAccess
                | VariablePointers
                | VariablePointersStorageBuffer
                | UniformTexelBufferArrayDynamicIndexing
                | StorageTexelBufferArrayDynamicIndexing
                | Int8
                | StorageBuffer8BitAccess
                | FloatControls2
                | SampleRateShading
                | StorageImageExtendedFormats
                | ImageGatherExtended
        )
}

fn is_support_optional_vulkan_1_0(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    matches!(
        capability,
        Geometry
            | Tessellation
            | Float64
            | Int64
            | Int16
            | TessellationPointSize
            | GeometryPointSize
            | ImageGatherExtended
            | StorageImageMultisample
            | UniformBufferArrayDynamicIndexing
            | SampledImageArrayDynamicIndexing
            | StorageBufferArrayDynamicIndexing
            | StorageImageArrayDynamicIndexing
            | ClipDistance
            | CullDistance
            | ImageCubeArray
            | SampleRateShading
            | SparseResidency
            | MinLod
            | SampledCubeArray
            | ImageMSArray
            | StorageImageExtendedFormats
            | InterpolationFunction
            | StorageImageReadWithoutFormat
            | StorageImageWriteWithoutFormat
            | MultiViewport
            | Int64Atomics
            | TransformFeedback
            | GeometryStreams
            | Float16
            | Int8
            | BFloat16TypeKHR
            | Float8EXT
    )
}

fn is_support_optional_vulkan_1_1(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_optional_vulkan_1_0(capability)
        || matches!(
            capability,
            GroupNonUniform
                | GroupNonUniformVote
                | GroupNonUniformArithmetic
                | GroupNonUniformBallot
                | GroupNonUniformShuffle
                | GroupNonUniformShuffleRelative
                | GroupNonUniformClustered
                | GroupNonUniformQuad
                | DrawParameters
                | rspirv::spirv::Capability::StorageUniformBufferBlock16
                | rspirv::spirv::Capability::StorageUniform16
                | StoragePushConstant16
                | StorageInputOutput16
                | DeviceGroup
                | MultiView
                | VariablePointersStorageBuffer
                | VariablePointers
        )
}

fn is_support_optional_vulkan_1_2(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_optional_vulkan_1_1(capability)
        || matches!(
            capability,
            DenormPreserve
                | DenormFlushToZero
                | SignedZeroInfNanPreserve
                | RoundingModeRTE
                | RoundingModeRTZ
                | VulkanMemoryModel
                | VulkanMemoryModelDeviceScope
                | StorageBuffer8BitAccess
                | UniformAndStorageBuffer8BitAccess
                | StoragePushConstant8
                | ShaderViewportIndex
                | ShaderLayer
                | PhysicalStorageBufferAddresses
                | RuntimeDescriptorArray
                | UniformTexelBufferArrayDynamicIndexing
                | StorageTexelBufferArrayDynamicIndexing
                | UniformBufferArrayNonUniformIndexing
                | SampledImageArrayNonUniformIndexing
                | StorageBufferArrayNonUniformIndexing
                | StorageImageArrayNonUniformIndexing
                | InputAttachmentArrayNonUniformIndexing
                | UniformTexelBufferArrayNonUniformIndexing
                | StorageTexelBufferArrayNonUniformIndexing
        )
}

fn is_support_optional_vulkan_1_3(capability: rspirv::spirv::Capability) -> bool {
    is_support_optional_vulkan_1_2(capability)
}

fn is_support_optional_vulkan_1_4(capability: rspirv::spirv::Capability) -> bool {
    is_support_optional_vulkan_1_3(capability)
}

fn is_support_optional_opencl_1_2(capability: rspirv::spirv::Capability) -> bool {
    use rspirv::spirv::Capability::*;
    matches!(capability, ImageBasic | Float64 | Float16)
}

fn is_support_guaranteed_opencl_1_2(capability: rspirv::spirv::Capability, embedded: bool) -> bool {
    use rspirv::spirv::Capability::*;
    matches!(
        capability,
        Addresses | Float16Buffer | Int16 | Int8 | Kernel | Linkage | Vector16
    ) || (!embedded && capability == Int64)
}

fn is_support_guaranteed_opencl_2_0(capability: rspirv::spirv::Capability, embedded: bool) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_guaranteed_opencl_1_2(capability, embedded)
        || matches!(capability, DeviceEnqueue | GenericPointer | Groups | Pipes)
}

fn is_support_guaranteed_opencl_2_2(capability: rspirv::spirv::Capability, embedded: bool) -> bool {
    use rspirv::spirv::Capability::*;
    is_support_guaranteed_opencl_2_0(capability, embedded)
        || matches!(capability, SubgroupDispatch | PipeStorage)
}

fn is_opencl_capability_allowed(env: TargetEnv, capability: rspirv::spirv::Capability) -> bool {
    let embedded = matches!(
        env,
        TargetEnv::OpenClEmbedded1_2
            | TargetEnv::OpenClEmbedded2_0
            | TargetEnv::OpenClEmbedded2_1
            | TargetEnv::OpenClEmbedded2_2
    );
    match env {
        TargetEnv::OpenCl1_2 | TargetEnv::OpenClEmbedded1_2 => {
            is_support_guaranteed_opencl_1_2(capability, embedded)
                || is_support_optional_opencl_1_2(capability)
        }
        TargetEnv::OpenCl2_0
        | TargetEnv::OpenClEmbedded2_0
        | TargetEnv::OpenCl2_1
        | TargetEnv::OpenClEmbedded2_1 => {
            is_support_guaranteed_opencl_2_0(capability, embedded)
                || is_support_optional_opencl_1_2(capability)
        }
        TargetEnv::OpenCl2_2 | TargetEnv::OpenClEmbedded2_2 => {
            is_support_guaranteed_opencl_2_2(capability, embedded)
                || is_support_optional_opencl_1_2(capability)
        }
        _ => false,
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

    #[test]
    fn parse_name_matches_prefixes() {
        assert_eq!(
            TargetEnv::parse_name("vulkan1.2"),
            Some(TargetEnv::Vulkan1_2)
        );
        assert_eq!(
            TargetEnv::parse_name("opencl2.2embedded"),
            Some(TargetEnv::OpenClEmbedded2_2)
        );
        assert_eq!(TargetEnv::parse_name("unknown"), None);
    }

    #[test]
    fn parse_vulkan_env_respects_capabilities() {
        let env = TargetEnv::parse_vulkan_env((1 << 22) | (2 << 12), 0x10500);
        assert_eq!(env, Some(TargetEnv::Vulkan1_2));
        let impossible = TargetEnv::parse_vulkan_env(2 << 22, 0x10700);
        assert_eq!(impossible, None);
    }

    #[test]
    fn listing_is_non_empty() {
        let list = TargetEnv::list_target_envs(4, 40);
        assert!(list.contains("vulkan1.0"));
        assert!(list.contains("|"));
    }

    #[test]
    fn intel_function_variants_extension_allowlist() {
        use super::ExtensionName;
        let ext = ExtensionName::from("SPV_INTEL_function_variants");
        assert!(!TargetEnv::Vulkan1_2.is_extension_allowed(&ext));
        assert!(TargetEnv::OpenCl2_2.is_extension_allowed(&ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&ext));
        assert!(!TargetEnv::WebGpu0.is_extension_allowed(&ext));
    }

    #[test]
    fn vendor_extensions_gate_by_environment() {
        use super::ExtensionName;
        let nv_ext = ExtensionName::from("SPV_NV_mesh_shader");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&nv_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&nv_ext));
        assert!(!TargetEnv::OpenCl2_2.is_extension_allowed(&nv_ext));

        let nvx_ext = ExtensionName::from("SPV_NVX_multiview_per_view_attributes");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&nvx_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&nvx_ext));
        assert!(!TargetEnv::OpenCl2_2.is_extension_allowed(&nvx_ext));

        let amd_ext = ExtensionName::from("SPV_AMD_shader_trinary_minmax");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&amd_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&amd_ext));
        assert!(!TargetEnv::OpenCl1_2.is_extension_allowed(&amd_ext));
        assert!(!TargetEnv::OpenGl4_5.is_extension_allowed(&amd_ext));

        let amdx_ext = ExtensionName::from("SPV_AMDX_shader_enqueue");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&amdx_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&amdx_ext));
        assert!(!TargetEnv::OpenCl1_2.is_extension_allowed(&amdx_ext));

        let google_ext = ExtensionName::from("SPV_GOOGLE_decorate_string");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&google_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&google_ext));
        assert!(!TargetEnv::OpenCl1_2.is_extension_allowed(&google_ext));
        assert!(!TargetEnv::OpenGl4_5.is_extension_allowed(&google_ext));

        let qcom_ext = ExtensionName::from("SPV_QCOM_image_processing");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&qcom_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&qcom_ext));
        assert!(!TargetEnv::OpenCl1_2.is_extension_allowed(&qcom_ext));

        let arm_ext = ExtensionName::from("SPV_ARM_core_builtins");
        assert!(TargetEnv::Vulkan1_2.is_extension_allowed(&arm_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&arm_ext));
        assert!(!TargetEnv::OpenCl2_2.is_extension_allowed(&arm_ext));
    }

    #[test]
    fn opencl_vendor_extensions_are_blocked_for_vulkan() {
        use super::ExtensionName;
        let altera_ext = ExtensionName::from("SPV_ALTERA_fpga_memory_attributes");
        assert!(TargetEnv::OpenCl2_2.is_extension_allowed(&altera_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&altera_ext));
        assert!(!TargetEnv::Vulkan1_2.is_extension_allowed(&altera_ext));

        let intel_ext = ExtensionName::from("SPV_INTEL_fpga_reg");
        assert!(TargetEnv::OpenCl2_1.is_extension_allowed(&intel_ext));
        assert!(TargetEnv::Universal1_5.is_extension_allowed(&intel_ext));
        assert!(!TargetEnv::Vulkan1_1.is_extension_allowed(&intel_ext));
    }

    #[test]
    fn webgpu_rejects_all_extensions() {
        use super::ExtensionName;
        let nv_ext = ExtensionName::from("SPV_NV_mesh_shader");
        let amd_ext = ExtensionName::from("SPV_AMD_shader_trinary_minmax");
        assert!(!TargetEnv::WebGpu0.is_extension_allowed(&nv_ext));
        assert!(!TargetEnv::WebGpu0.is_extension_allowed(&amd_ext));
    }

    #[test]
    fn opencl_accepts_intel_vendor_extensions() {
        use super::ExtensionName;
        let intel_ext = ExtensionName::from("SPV_INTEL_shader_integer_functions2");
        assert!(TargetEnv::OpenCl2_2.is_extension_allowed(&intel_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&intel_ext));
        assert!(!TargetEnv::Vulkan1_2.is_extension_allowed(&intel_ext));
        assert!(!TargetEnv::OpenGl4_5.is_extension_allowed(&intel_ext));
    }

    #[test]
    fn opencl_rejects_general_vendor_extensions() {
        use super::ExtensionName;
        let google_ext = ExtensionName::from("SPV_GOOGLE_hlsl_functionality1");
        assert!(!TargetEnv::OpenCl2_2.is_extension_allowed(&google_ext));
        assert!(TargetEnv::Universal1_6.is_extension_allowed(&google_ext));
        assert!(!TargetEnv::OpenGl4_5.is_extension_allowed(&google_ext));
    }
}
