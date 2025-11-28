#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use spirv_tools_opt::{fuzzing::arbitrary_expr, optimize_expr};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(expr) = arbitrary_expr(&mut u) {
        let _ = optimize_expr(&expr);
    }
});
