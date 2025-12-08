use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{default_fuzz_options, fuzz_module, fuzz_module_with_cpp, validate_binary};

fn minimal_words() -> Vec<u32> {
    assemble_text(
        "\
OpCapability Shader
OpMemoryModel Logical GLSL450
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
OpEntryPoint Vertex %main \"main\"
",
    )
    .expect("assemble")
}

#[test]
fn cpp_fuzz_bridge_validates_or_skips() {
    let words = minimal_words();
    let opts = default_fuzz_options();
    let cpp = fuzz_module_with_cpp(&words, &opts);
    if !cpp.success {
        eprintln!("C++ fuzz bridge unavailable or disabled: {}", cpp.message);
        return;
    }
    assert!(
        validate_binary(TargetEnv::Universal1_6, &cpp.words).success,
        "C++ fuzz bridge should emit a valid module"
    );
}

#[test]
fn rust_and_cpp_fuzz_both_succeed_when_cpp_available() {
    let words = minimal_words();
    let opts = default_fuzz_options();
    let cpp = fuzz_module_with_cpp(&words, &opts);
    if !cpp.success {
        eprintln!("C++ fuzz bridge unavailable or disabled: {}", cpp.message);
        return;
    }

    let rust = fuzz_module(&words);
    assert!(rust.success, "Rust fuzz pipeline should succeed");
    assert!(cpp.success, "C++ fuzz bridge should succeed");
    assert!(
        validate_binary(TargetEnv::Universal1_6, &rust.words).success,
        "Rust fuzz pipeline should emit a valid module"
    );
    assert!(
        validate_binary(TargetEnv::Universal1_6, &cpp.words).success,
        "C++ fuzz bridge should emit a valid module"
    );
}

#[test]
fn cpp_and_rust_fuzz_match_on_corpus_when_cpp_available() {
    let corpus = vec![
        minimal_words(),
        assemble_text(
            "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
        )
        .expect("assemble fragment"),
        assemble_text(
            "\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
",
        )
        .expect("assemble compute"),
    ];

    let opts = default_fuzz_options();
    let mut cpp_unavailable = false;
    for (idx, words) in corpus.iter().enumerate() {
        let cpp = fuzz_module_with_cpp(words, &opts);
        if !cpp.success {
            eprintln!(
                "C++ fuzz bridge unavailable or disabled on corpus {idx}: {}",
                cpp.message
            );
            cpp_unavailable = true;
            break;
        }
        let rust = fuzz_module(words);
        assert!(rust.success, "Rust fuzz pipeline failed on corpus {idx}");
        assert!(cpp.success, "C++ fuzz pipeline failed on corpus {idx}");
        assert_eq!(
            rust.words, cpp.words,
            "Rust and C++ fuzz outputs diverged on corpus {idx}"
        );
    }

    if cpp_unavailable {
        eprintln!("Skipping corpus parity because C++ fuzz bridge is unavailable");
    }
}
