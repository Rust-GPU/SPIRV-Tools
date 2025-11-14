use spirv_tools_core::TargetEnv;

#[cxx::bridge(namespace = "spvtools::ffi")]
mod ffi {
    extern "Rust" {
        fn describe_target_env(env: u32) -> String;
    }
}

/// Returns the human-readable description for a SPIR-V target environment.
pub fn describe_target_env(env: u32) -> String {
    TargetEnv::from_raw(env)
        .map(TargetEnv::description)
        .unwrap_or("")
        .to_string()
}
