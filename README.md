**This repo was a prototype for
[BLAKE3](https://github.com/BLAKE3-team/BLAKE3), which was announced on
9 January 2020.**

# Baokeshed [![Actions Status](https://github.com/oconnor663/baokeshed/workflows/tests/badge.svg)](https://github.com/oconnor663/baokeshed/actions)

To build the command line utility:

```
cd baokeshed_bin
cargo build --release
./target/release/baokeshed --help
```

Other useful commands:

```
cargo doc --open
cargo test
cargo +nightly bench

# Benchmark the Rayon-based multithreaded implementation in Rust
cargo +nightly bench --features=rayon

# Benchmark SSE4.1, AVX2, and AVX512 implementations in C.
cargo +nightly bench --features=c_detect

# Benchmark the NEON implementation in C.
cargo +nightly bench --features=c_neon
```
