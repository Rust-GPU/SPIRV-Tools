use super::super::*;

#[test]
fn valid_module_cache_reuses_entries() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble");
    let mut cache = ValidModuleCache::default();
    let first = cache
        .validate_words(&binary, TargetEnv::Universal1_6)
        .expect("first validation");
    let second = cache
        .validate_words(&binary, TargetEnv::Universal1_6)
        .expect("cached validation");
    assert_eq!(
        Arc::as_ptr(&first),
        Arc::as_ptr(&second),
        "cached entries should reuse the same allocation"
    );
}
