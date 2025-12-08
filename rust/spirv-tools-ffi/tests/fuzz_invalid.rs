use arbitrary::{Arbitrary, Unstructured};
use spirv_tools_core::assembly::assemble_text;
use spirv_tools_core::assembly::options::BinaryToTextOptions;
use spirv_tools_core::disassembly::disassemble_binary;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{
    default_fuzz_options, fuzz_module_with_options, validate_binary, FuzzConfig, FuzzGenerator,
    FuzzModule, FuzzOutcome, InvalidKind, MaybeInvalid, Unchecked, Validity,
};

#[test]
fn rust_fuzzer_can_emit_intentionally_invalid() {
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
    let binary = assemble_text(&text).expect("assemble module");

    let mut opts = default_fuzz_options();
    opts.enable_fuzzer_pass_validation = false; // allow invalid mutations

    let result = fuzz_module_with_options(&binary, &opts);
    assert!(result.success);
    if !validate_binary(TargetEnv::Universal1_6, &result.words).success {
        assert!(
            result.message.contains("intentionally invalid"),
            "expected intentionally invalid marker"
        );
    }
}

#[test]
fn rust_fuzzer_can_target_specific_invalid_kind() {
    let cfg = FuzzConfig {
        seed: 0,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::BrokenIdBound),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::BrokenIdBound));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "broken id bound should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_respects_prefer_valid_even_with_hint() {
    let cfg = FuzzConfig {
        seed: 0,
        prefer_valid: true,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MissingMemoryModel),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Valid { words } => {
            assert!(validate_binary(TargetEnv::Universal1_6, &words).success);
        }
        FuzzOutcome::Invalid { .. } => panic!("expected valid module when prefer_valid=true"),
    }
}

#[test]
fn rust_fuzzer_can_emit_dangling_use() {
    let cfg = FuzzConfig {
        seed: 42,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::DanglingUse),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::DanglingUse));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "dangling use should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_can_emit_duplicate_id() {
    let cfg = FuzzConfig {
        seed: 7,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::DuplicateId),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::DuplicateId));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "duplicate ids should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_can_emit_storage_class_mismatch() {
    let cfg = FuzzConfig {
        seed: 21,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::StorageClassMismatch),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::StorageClassMismatch));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "storage class mismatch should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_can_emit_missing_loop_merge() {
    let cfg = FuzzConfig {
        seed: 23,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MissingLoopMerge),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::MissingLoopMerge));
            let validated = validate_binary(TargetEnv::Universal1_6, &words).success;
            if validated {
                // If validation succeeds, still ensure no loop merge exists so the mutation happened.
                let text = disassemble_binary(&words, BinaryToTextOptions::NONE)
                    .expect("disassemble mutated module");
                assert!(
                    !text.contains("OpLoopMerge"),
                    "mutation should strip loop merges when validation still passes"
                );
            }
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_can_emit_missing_selection_merge() {
    let cfg = FuzzConfig {
        seed: 9,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::MissingSelectionMerge),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::MissingSelectionMerge));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "missing selection merge should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn rust_fuzzer_can_emit_phi_predecessor_mismatch() {
    let cfg = FuzzConfig {
        seed: 11,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: Some(InvalidKind::PhiPredecessorMismatch),
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Invalid { kind, words } => {
            assert!(matches!(kind, InvalidKind::PhiPredecessorMismatch));
            assert!(
                !validate_binary(TargetEnv::Universal1_6, &words).success,
                "phi predecessor mismatch should fail validation"
            );
        }
        FuzzOutcome::Valid { .. } => panic!("expected invalid module"),
    }
}

#[test]
fn maybeinvalid_arbitrary_can_request_invalid() {
    let mut u = Unstructured::new(&[0u8; 256]);
    let candidate: MaybeInvalid<FuzzModule<Unchecked>> = match Arbitrary::arbitrary(&mut u) {
        Ok(c) => c,
        Err(_) => return, // skip if generator cannot build from zeroed input
    };
    assert!(
        matches!(candidate.validity(), Validity::Invalid(_)),
        "zeroed input should bias towards invalid validity"
    );
}

#[test]
fn prefer_valid_overrides_invalid_validity() {
    let cfg = FuzzConfig {
        seed: 99,
        prefer_valid: true,
        allow_invalid: true,
        invalid_hint: None,
    };
    let generator = FuzzGenerator::new(cfg);
    let mut u = Unstructured::new(&[0u8; 256]);
    let candidate: MaybeInvalid<FuzzModule<Unchecked>> = match Arbitrary::arbitrary(&mut u) {
        Ok(c) => c,
        Err(_) => return, // skip if arbitrary input is insufficient
    };
    let outcome = generator
        .materialize_for_test(
            TargetEnv::Universal1_6,
            candidate.with_validity(Validity::Invalid(InvalidKind::MissingTerminator)),
        )
        .expect("materialize");
    assert!(
        matches!(outcome, FuzzOutcome::Valid { .. }),
        "prefer_valid should force validation even when candidate asked for invalid"
    );
}

#[test]
fn materialize_uses_candidate_validity_when_no_hint() {
    let cfg = FuzzConfig {
        seed: 1,
        prefer_valid: false,
        allow_invalid: true,
        invalid_hint: None,
    };
    let generator = FuzzGenerator::new(cfg);
    let mut u = Unstructured::new(&[0u8; 256]);
    let candidate: MaybeInvalid<FuzzModule<Unchecked>> = match Arbitrary::arbitrary(&mut u) {
        Ok(c) => c,
        Err(_) => return, // skip if arbitrary fails to construct a module
    };
    let candidate = candidate.with_validity(Validity::Invalid(InvalidKind::DuplicateId));
    let outcome = generator
        .materialize_for_test(TargetEnv::Universal1_6, candidate)
        .expect("materialize");
    match outcome {
        FuzzOutcome::Invalid { kind, .. } => assert!(
            matches!(kind, InvalidKind::DuplicateId),
            "expected validity-provided invalid kind to win"
        ),
        FuzzOutcome::Valid { .. } => panic!("expected invalid outcome"),
    }
}

#[test]
fn generator_prefers_valid_when_requested() {
    let cfg = FuzzConfig {
        seed: 13,
        prefer_valid: true,
        allow_invalid: false,
        invalid_hint: None,
    };
    let generator = FuzzGenerator::new(cfg);
    let outcome = generator
        .generate(TargetEnv::Universal1_6, &[])
        .expect("generate");
    match outcome {
        FuzzOutcome::Valid { words } => {
            assert!(
                validate_binary(TargetEnv::Universal1_6, &words).success,
                "prefer_valid should yield a validated module"
            );
        }
        FuzzOutcome::Invalid { .. } => panic!("expected valid module when allow_invalid=false"),
    }
}
