fn main() {
    cc::Build::new()
        .file("c/baokeshed.c")
        .flag("-O3")
        .flag("-march=native")
        .compile("cbaokeshed");
}
