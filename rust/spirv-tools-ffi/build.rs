fn main() {
    cxx_build::bridge("src/lib.rs")
        .std("c++17")
        .compile("spirv-tools-ffi");
}
