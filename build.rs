use std::env;

fn defined(name: &str) -> bool {
    env::var_os(name).is_some()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !defined("CARGO_FEATURE_C_PORTABLE") {
        return Ok(());
    }

    let target = env::var("TARGET")?;
    let target_components: Vec<&str> = target.split("-").collect();
    let target_arch = target_components[0];
    let target_os = target_components[2];
    let is_x86 = target_arch == "x86_64" || target_arch == "i686";
    let is_windows = target_os == "windows";

    // Note that under -march=native, Clang seems to perform better than GCC.
    // To keep this build portable, we use the system default compiler. This
    // builder respects the $CC env var to override
    let mut build = cc::Build::new();
    build.file("c/baokeshed.c");
    build.file("c/baokeshed_portable.c");
    if is_x86 {
        build.file("c/baokeshed_sse41.c");
        build.file("c/baokeshed_avx2.c");
        build.file("c/baokeshed_avx512.c");
    }
    if defined("CARGO_FEATURE_C_SSE41") {
        if is_windows {
            // /arch:SSE2 is the default on x86 and undefined on x86_64:
            // https://docs.microsoft.com/en-us/cpp/build/reference/arch-x86
            // It also includes SSE4.1 intrisincs:
            // https://stackoverflow.com/a/32183222/823869
            // But we do need to explicitly define the __SSE4_1__ constant:
            // https://stackoverflow.com/a/18566249/823869
            build.flag("/D__SSE4_1__");
        } else {
            build.flag("-msse4.1");
        }
    }
    if defined("CARGO_FEATURE_C_AVX2") {
        if is_windows {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2");
        }
    }
    if defined("CARGO_FEATURE_C_AVX512") {
        if is_windows {
            build.flag("/arch:AVX512");
        } else {
            build.flag("-mavx512f");
            build.flag("-mavx512vl");
        }
    }
    if defined("CARGO_FEATURE_C_NATIVE") {
        if is_windows {
            panic!("Windows doesn't have an equivalent of -march=native.");
        } else {
            build.flag("-march=native");
        }
    }
    build.compile("cbaokeshed");

    // The `cc` crate does not automatically emit rerun-if directives for the
    // environment variables it supports, in particular for $CC. We expect to
    // do a lot of benchmarking across different compilers, so we explicitly
    // add the variables that we're likely to need.
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=CFLAGS");

    // Ditto for source files, though these shouldn't change as often.
    for file in std::fs::read_dir("c")? {
        println!(
            "cargo:rerun-if-changed={}",
            file?.path().to_str().expect("utf-8")
        );
    }

    Ok(())
}
