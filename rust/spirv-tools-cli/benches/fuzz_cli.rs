use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};
use spirv_tools_cli::assembly::words_to_bytes;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn build_minimal_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
    let void = b.type_void();
    let fn_ty = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, FunctionControl::NONE, fn_ty)
        .expect("begin function");
    b.begin_block(None).expect("begin block");
    b.ret().expect("ret");
    b.end_function().expect("end function");
    b.entry_point(rspirv::spirv::ExecutionModel::Vertex, func, "main", []);
    b.module().assemble()
}

fn write_module(tempdir: &TempDir, words: &[u32]) -> PathBuf {
    let path = tempdir.path().join("module.spv");
    std::fs::write(&path, words_to_bytes(words)).expect("write module");
    path
}

fn bench_fuzz_cli(c: &mut Criterion) {
    let rust_bin = std::env::var_os("CARGO_BIN_EXE_spirv-fuzz")
        .or_else(|| which::which("spirv-fuzz").ok().map(|p| p.into_os_string()));
    let Some(rust_bin) = rust_bin else {
        eprintln!("spirv-fuzz binary not built or not found on PATH; skipping fuzz bench");
        return;
    };

    let module = build_minimal_module();
    let tempdir = TempDir::new().expect("tempdir");
    let module_path = write_module(&tempdir, &module);

    let mut group = c.benchmark_group("fuzz-cli");
    group.bench_with_input(
        BenchmarkId::new("rust", "valid"),
        &module_path,
        |b, path| {
            b.iter(|| {
                let out = tempdir.path().join("rust-out.spv");
                let status = Command::new(&rust_bin)
                    .arg(path)
                    .arg("-o")
                    .arg(&out)
                    .status()
                    .expect("run rust spirv-fuzz");
                assert!(status.success(), "rust spirv-fuzz failed: {status:?}");
            });
        },
    );

    if let Some(cpp_bin) = std::env::var_os("SPIRV_CPP_FUZZ")
        .or_else(|| which::which("spirv-fuzz").ok().map(|p| p.into_os_string()))
    {
        group.bench_with_input(BenchmarkId::new("cpp", "valid"), &module_path, |b, path| {
            b.iter(|| {
                let out = tempdir.path().join("cpp-out.spv");
                let status = Command::new(&cpp_bin)
                    .arg(path)
                    .arg("-o")
                    .arg(&out)
                    .status()
                    .expect("run cpp spirv-fuzz");
                assert!(status.success(), "cpp spirv-fuzz failed: {status:?}");
            });
        });
    } else {
        eprintln!("SPIRV_CPP_FUZZ not set and spirv-fuzz not on PATH; skipping cpp bench");
    }

    group.finish();
}

criterion_group!(benches, bench_fuzz_cli);
criterion_main!(benches);
