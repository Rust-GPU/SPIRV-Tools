use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use spirv_tools_cli::disassemble::InputSource;
use spirv_tools_cli::optimizer::{run_optimize, OptimizeConfig};
use std::path::PathBuf;
use tempfile::TempDir;

fn build_const_add_module() -> Vec<u32> {
    use rspirv::dr::Builder;
    use rspirv::spirv::{AddressingModel, Capability, FunctionControl, MemoryModel};

    let mut b = Builder::new();
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::Simple);
    let void = b.type_void();
    let int = b.type_int(32, 0);
    let func_ty = b.type_function(void, vec![]);
    let _ = b
        .begin_function(void, None, FunctionControl::NONE, func_ty)
        .expect("function");
    let _ = b.begin_block(None).expect("block");
    let c2 = b.constant_bit32(int, 2);
    let c3 = b.constant_bit32(int, 3);
    let _add = b.i_add(int, None, c2, c3).expect("add");
    b.ret().expect("ret");
    b.end_function().expect("end");
    b.module().assemble()
}

fn write_module(tempdir: &TempDir, words: &[u32]) -> PathBuf {
    let path = tempdir.path().join("module.spv");
    let bytes = spirv_tools_cli::assembly::words_to_bytes(words);
    std::fs::write(&path, &bytes).expect("write module");
    path
}

fn bench_optimizer_cli(c: &mut Criterion) {
    let module = build_const_add_module();
    let tempdir = TempDir::new().expect("tempdir");
    let module_path = write_module(&tempdir, &module);

    let mut group = c.benchmark_group("optimizer-cli");
    let rust_config = OptimizeConfig {
        input: InputSource::Path(module_path.clone()),
        output: None,
        rust_arith_pass: true,
        cpp_opt_path: None,
        force_rust_opt: true,
    };
    group.bench_with_input(BenchmarkId::new("rust", "add"), &rust_config, |b, cfg| {
        b.iter(|| run_optimize(cfg).expect("optimize"));
    });

    let passthrough_config = OptimizeConfig {
        rust_arith_pass: false,
        ..rust_config.clone()
    };
    group.bench_with_input(
        BenchmarkId::new("passthrough", "add"),
        &passthrough_config,
        |b, cfg| {
            b.iter(|| run_optimize(cfg).expect("passthrough"));
        },
    );

    if let Some(cpp_opt) = std::env::var_os("SPIRV_CPP_OPT").or_else(|| {
        std::env::var_os("PATH")
            .and_then(|_| which::which("spirv-opt").ok().map(|p| p.into_os_string()))
    }) {
        let cpp_config = OptimizeConfig {
            rust_arith_pass: false,
            cpp_opt_path: Some(cpp_opt),
            ..rust_config
        };
        group.bench_with_input(BenchmarkId::new("cpp", "add"), &cpp_config, |b, cfg| {
            b.iter(|| run_optimize(cfg).expect("cpp optimize"));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_optimizer_cli);
criterion_main!(benches);
