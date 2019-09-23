fn main() {
    cc::Build::new()
        .file("c/baokeshed.c")
        .file("c/baokeshed_portable.c")
        .flag("-O3")
        .flag("-march=native")
        .compile("cbaokeshed");
}
