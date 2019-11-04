use std::env;

const C_SSE41_VAR: &str = "CARGO_FEATURE_C_SSE41";
const C_AVX2_VAR: &str = "CARGO_FEATURE_C_AVX2";
const C_AVX512_VAR: &str = "CARGO_FEATURE_C_AVX512";
const C_ARMV7NEON_VAR: &str = "CARGO_FEATURE_C_ARMV7NEON";

fn defined(var: &str) -> bool {
    env::var_os(var).is_some()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn is_avx512_detected() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
}

// Set env vars to signal the rest of this build script, and emit cargo
// directives to turn on feature during crate compilation. Note that this can't
// activate feature dependencies, so we rely on the fact that e.g. SSE4.1 is
// always detected when AVX2 is detected.
#[allow(unreachable_code)]
fn detect_c_features() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse4.1") {
            env::set_var(C_SSE41_VAR, "1");
            println!(r#"cargo:rustc-cfg=feature="c_sse41""#);
        }
        if is_x86_feature_detected!("avx2") {
            env::set_var(C_AVX2_VAR, "1");
            println!(r#"cargo:rustc-cfg=feature="c_avx2""#);
        }
        if is_avx512_detected() {
            env::set_var(C_AVX512_VAR, "1");
            println!(r#"cargo:rustc-cfg=feature="c_avx512""#);
        }
        return;
    }
    panic!("c_detect not supported on non-x86");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !defined("CARGO_FEATURE_C_PORTABLE") {
        return Ok(());
    }

    if defined("CARGO_FEATURE_C_DETECT") {
        detect_c_features();
    }

    let target = env::var("TARGET")?;
    let target_components: Vec<&str> = target.split("-").collect();
    let target_arch = target_components[0];
    let is_armv7 = target_arch == "armv7";
    let target_os = target_components[2];
    let is_windows = target_os == "windows";

    // Note that under -march=native, Clang seems to perform better than GCC.
    // To keep this build portable, we use the system default compiler. This
    // builder respects the $CC env var to override
    let mut build = cc::Build::new();
    build.file("c/baokeshed.c");
    build.file("c/baokeshed_portable.c");
    if !is_windows {
        build.flag("-std=c11");
    }
    if defined(C_SSE41_VAR) {
        build.file("c/baokeshed_sse41.c");
        build.define("BAOKESHED_USE_SSE41", "1");
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
    if defined(C_AVX2_VAR) {
        build.file("c/baokeshed_avx2.c");
        build.define("BAOKESHED_USE_AVX2", "1");
        if is_windows {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2");
        }
    }
    if defined(C_AVX512_VAR) {
        build.file("c/baokeshed_avx512.c");
        build.define("BAOKESHED_USE_AVX512", "1");
        if is_windows {
            // Note that a lot of versions of MSVC don't support /arch:AVX512,
            // and they'll discard it with a warning, giving you an AVX2 build.
            build.flag("/arch:AVX512");
        } else {
            build.flag("-mavx512f");
            build.flag("-mavx512vl");
        }
    }
    if defined(C_ARMV7NEON_VAR) {
        build.file("c/baokeshed_neon.c");
        build.define("BAOKESHED_USE_NEON", "1");
        // Note that AArch64 supports NEON by default and does not support -mpfu.
        if is_armv7 {
            // Match https://github.com/BLAKE2/BLAKE2/blob/master/neon/makefile#L2.
            build.flag("-mfpu=neon-vfpv4");
            build.flag("-mfloat-abi=hard");
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
