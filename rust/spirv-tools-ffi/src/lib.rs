use spirv_tools_core::TargetEnv;

#[cxx::bridge(namespace = "spvtools::ffi")]
mod ffi {
    #[derive(Debug)]
    struct ParseResult {
        success: bool,
        env: u32,
    }

    extern "Rust" {
        fn describe_target_env(env: u32) -> String;
        fn spirv_version_for_target(env: u32) -> u32;
        fn parse_target_env(input: &str) -> ParseResult;
        fn parse_vulkan_env(vulkan_version: u32, spirv_version: u32) -> ParseResult;
        fn read_env_from_text(text: &[u8]) -> ParseResult;
        fn is_vulkan_env(env: u32) -> bool;
        fn is_opencl_env(env: u32) -> bool;
        fn is_opengl_env(env: u32) -> bool;
        fn is_valid_env(env: u32) -> bool;
        fn log_namespace(env: u32) -> String;
        fn list_target_envs(pad: usize, wrap: usize) -> String;
    }
}

/// Returns the human-readable description for a SPIR-V target environment.
pub fn describe_target_env(env: u32) -> String {
    TargetEnv::from_raw(env)
        .map(TargetEnv::description)
        .unwrap_or("")
        .to_string()
}

fn to_env(value: u32) -> Option<TargetEnv> {
    TargetEnv::from_raw(value)
}

pub fn spirv_version_for_target(env: u32) -> u32 {
    to_env(env)
        .map(|env| env.spirv_version().to_word())
        .unwrap_or_default()
}

pub fn parse_target_env(input: &str) -> ffi::ParseResult {
    match TargetEnv::parse_name(input) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn parse_vulkan_env(vulkan_version: u32, spirv_version: u32) -> ffi::ParseResult {
    match TargetEnv::parse_vulkan_env(vulkan_version, spirv_version) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn read_env_from_text(text: &[u8]) -> ffi::ParseResult {
    match spirv_tools_core::target_env::read_env_from_text(text) {
        Some(env) => ffi::ParseResult {
            success: true,
            env: env.to_raw(),
        },
        None => ffi::ParseResult {
            success: false,
            env: TargetEnv::Universal1_0.to_raw(),
        },
    }
}

pub fn is_vulkan_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_vulkan()).unwrap_or(false)
}

pub fn is_opencl_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_opencl()).unwrap_or(false)
}

pub fn is_opengl_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_opengl()).unwrap_or(false)
}

pub fn is_valid_env(env: u32) -> bool {
    to_env(env).map(|env| env.is_valid()).unwrap_or(false)
}

pub fn log_namespace(env: u32) -> String {
    to_env(env)
        .map(|env| env.log_namespace().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

pub fn list_target_envs(pad: usize, wrap: usize) -> String {
    TargetEnv::list_target_envs(pad, wrap)
}
