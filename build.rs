fn main() {
    // Using clang is a more apples-to-apples comparison with rustc, because
    // both are LLVM-based.
    std::env::set_var("CC", "clang");
    cc::Build::new()
        .file("c/baokeshed.c")
        .file("c/baokeshed_portable.c")
        .file("c/baokeshed_sse41.c")
        .file("c/baokeshed_avx2.c")
        .flag("-O3")
        .flag("-march=native")
        .compile("cbaokeshed");
}
