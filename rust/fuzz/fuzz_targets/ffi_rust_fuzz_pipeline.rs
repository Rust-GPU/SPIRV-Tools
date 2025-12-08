#![no_main]

use libfuzzer_sys::fuzz_target;
use spirv_tools_core::TargetEnv;
use spirv_tools_ffi::{
    default_fuzz_options, fuzz_module_with_options, validate_binary, FuzzConfig, FuzzGenerator,
    FuzzOutcome, InvalidKind,
};

fn decode_config(data: &[u8]) -> (FuzzConfig, &[u8]) {
    if data.is_empty() {
        return (
            FuzzConfig {
                seed: 0,
                prefer_valid: true,
                allow_invalid: false,
                invalid_hint: None,
            },
            data,
        );
    }
    let header = data[0];
    let allow_invalid = header & 0x1 != 0;
    let prefer_valid = header & 0x2 != 0;
    let hint = match (header >> 2) & 0x3 {
        0 => Some(InvalidKind::MissingMemoryModel),
        1 => Some(InvalidKind::MissingTerminator),
        2 => Some(InvalidKind::MissingEntryPoint),
        _ => Some(InvalidKind::BrokenIdBound),
    };
    let mut seed_bytes = [0u8; 8];
    for (dst, src) in seed_bytes.iter_mut().zip(data.iter().skip(1)) {
        *dst = *src;
    }
    let seed = u64::from_le_bytes(seed_bytes);
    (
        FuzzConfig {
            seed,
            prefer_valid,
            allow_invalid,
            invalid_hint: hint,
        },
        &data[1..],
    )
}

fuzz_target!(|data: &[u8]| {
    let (cfg, tail) = decode_config(data);
    let generator = FuzzGenerator::new(cfg);
    if let Ok(outcome) = generator.generate(TargetEnv::Universal1_6, tail) {
        let words = match outcome {
            FuzzOutcome::Valid { words } => words,
            FuzzOutcome::Invalid { words, .. } => words,
        };
        // Exercise the full fuzz entry point and validator; ignore errors to let fuzzing continue.
        let _ = fuzz_module_with_options(&words, &default_fuzz_options());
        let _ = validate_binary(TargetEnv::Universal1_6, &words);
    }
});
