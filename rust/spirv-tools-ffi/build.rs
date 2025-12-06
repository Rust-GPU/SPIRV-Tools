use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../cxxbridge/spirv-tools-ffi/src/context_bridge.h");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let include_root = repo_root.join("include");
    let headers_root = repo_root
        .join("external")
        .join("spirv-headers")
        .join("include");
    let build_root = repo_root.join("build-rust");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let include_dir = out_dir
        .join("cxxbridge")
        .join("include")
        .join("spirv-tools-ffi")
        .join("src");
    fs::create_dir_all(&include_dir).expect("failed to create bridge include dir");
    fs::copy(
        "../cxxbridge/spirv-tools-ffi/src/context_bridge.h",
        include_dir.join("context_bridge.h"),
    )
    .expect("failed to copy context bridge header");

    cxx_build::bridge("src/lib.rs")
        .file("src/context_bridge.cc")
        .include(repo_root)
        .include(include_root)
        .include(headers_root)
        .include(&build_root)
        .std("c++17")
        .compile("spirv-tools-ffi");

    let core_lib_dir = build_root.join("source");
    let opt_lib_dir = core_lib_dir.join("opt");
    let reduce_lib_dir = core_lib_dir.join("reduce");
    println!("cargo:rustc-link-search=native={}", core_lib_dir.display());
    println!("cargo:rustc-link-search=native={}", opt_lib_dir.display());
    println!("cargo:rustc-link-search=native={}", reduce_lib_dir.display());
    println!("cargo:rustc-link-lib=static=SPIRV-Tools");
    println!("cargo:rustc-link-lib=static=SPIRV-Tools-opt");
    println!("cargo:rustc-link-lib=static=SPIRV-Tools-reduce");
}
