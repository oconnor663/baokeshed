fn defined(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn main() {
    if !defined("CARGO_FEATURE_C") {
        return;
    }

    // Note that under -march=native, Clang seems to perform better than GCC.
    // To keep this build portable, we use the system default compiler. This
    // builder respects the $CC env var to override
    let mut build = cc::Build::new();
    build.file("c/baokeshed.c");
    build.file("c/baokeshed_portable.c");
    build.file("c/baokeshed_sse41.c");
    build.file("c/baokeshed_avx2.c");
    if defined("CARGO_FEATURE_C_SSE41") {
        build.flag_if_supported("-msse4.1");
    }
    if defined("CARGO_FEATURE_C_AVX2") {
        build.flag_if_supported("-mavx2");
    }
    if defined("CARGO_FEATURE_C_NATIVE") {
        build.flag_if_supported("-march=native");
    }
    build.compile("cbaokeshed");

    // The `cc` crate does not automatically emit rerun-if directives for the
    // environment variables it supports, in particular for $CC. We expect to
    // do a lot of benchmarking across different compilers, so we explicitly
    // add the variables that we're likely to need.
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CFLAGS");
}
