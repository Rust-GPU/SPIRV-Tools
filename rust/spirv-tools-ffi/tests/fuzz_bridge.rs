use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module_with_options, validate_binary};
use rspirv::binary::parse_words;
use rspirv::dr::Loader;
use rspirv::spirv::Op;

#[test]
fn cpp_fuzz_bridge_runs_with_seed() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%aux = OpFunction %void None %fn",
        "%aux_entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble module");

    let mut options = default_fuzz_options();
    options.random_seed = 1;
    options.replay_range = 1;
    options.enable_fuzzer_pass_validation = true;

    let result = fuzz_module_with_options(&binary, &options);
    assert!(result.success, "fuzzing should succeed: {}", result.message);
    assert!(
        !result.words.is_empty(),
        "fuzzing should yield a non-empty module"
    );

    // The module should remain valid after the fuzz run.
    assert!(validate_binary(TargetEnv::Universal1_6, &result.words).success);

    // A Nop should have been injected.
    let mut loader = Loader::new();
    parse_words(&result.words, &mut loader).expect("parse fuzzed module");
    let nop_count = loader
        .module()
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .filter(|inst| inst.class.opcode == Op::Nop)
        .count();
    assert!(nop_count >= 1);
}

#[test]
fn fuzz_seed_changes_target_block() {
    let text = [
        "OpCapability Shader",
        "OpMemoryModel Logical GLSL450",
        "%void = OpTypeVoid",
        "%fn = OpTypeFunction %void",
        "%main = OpFunction %void None %fn",
        "%entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
        "%aux = OpFunction %void None %fn",
        "%aux_entry = OpLabel",
        "OpReturn",
        "OpFunctionEnd",
    ]
    .join("\n");
    let binary = assemble_text(&text).expect("assemble module");

    let mut opts1 = default_fuzz_options();
    opts1.random_seed = 1;
    let mut opts2 = default_fuzz_options();
    opts2.random_seed = 2;

    let a = fuzz_module_with_options(&binary, &opts1);
    let b = fuzz_module_with_options(&binary, &opts2);
    assert!(a.success && b.success);
    assert_ne!(a.words, b.words, "different seeds should produce different layouts");
}
