#![feature(test)]

extern crate test;

use arrayref::array_ref;
use baokeshed::*;
use rand::prelude::*;
use test::Bencher;

const MEDIUM: usize = MAX_SIMD_DEGREE * CHUNK_LEN;

const LONG: usize = 1 << 20; // 1 MiB

// This struct randomizes two things:
// 1. The actual bytes of input.
// 2. The page offset the input starts at.
pub struct RandomInput {
    buf: Vec<u8>,
    len: usize,
    offsets: Vec<usize>,
    offset_index: usize,
}

impl RandomInput {
    pub fn new(b: &mut Bencher, len: usize) -> Self {
        b.bytes += len as u64;
        let page_size: usize = page_size::get();
        let mut buf = vec![0u8; len + page_size];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut buf);
        let mut offsets: Vec<usize> = (0..page_size).collect();
        offsets.shuffle(&mut rng);
        Self {
            buf,
            len,
            offsets,
            offset_index: 0,
        }
    }

    pub fn get(&mut self) -> &[u8] {
        let offset = self.offsets[self.offset_index];
        self.offset_index += 1;
        if self.offset_index >= self.offsets.len() {
            self.offset_index = 0;
        }
        &self.buf[offset..][..self.len]
    }
}

#[bench]
fn bench_hash_01_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_02_medium(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MEDIUM);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_03_chunk(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_04_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hasher_01_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_02_medium(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MEDIUM);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_03_chunk(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_04_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher_01_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher_02_medium(b: &mut Bencher) {
    // The C code supports AVX512 and ARM NEON. It also doesn't do any
    // multithreading. Always use 16 chunks as the "medium" length for
    // benchmarking C.
    let mut input = RandomInput::new(b, 16 * CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher_03_chunk(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher_04_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher64_01_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = c64::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher64_02_medium(b: &mut Bencher) {
    // The C code supports AVX512 and ARM NEON. It also doesn't do any
    // multithreading. Always use 16 chunks as the "medium" length for
    // benchmarking C.
    let mut input = RandomInput::new(b, 16 * CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c64::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher64_03_chunk(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c64::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_ffihasher64_04_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, baokeshed::portable64::BLOCK_LEN);
    b.iter(|| {
        let mut hasher = c64::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_xof(b: &mut Bencher) {
    let hasher = Hasher::new();
    let mut xof = hasher.finalize_xof();
    b.bytes = xof.read().len() as u64;
    b.iter(|| xof.read());
}

#[bench]
fn bench_compress_portable(b: &mut Bencher) {
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| {
        benchmarks::compress_portable(
            &mut state,
            array_ref!(input.get(), 0, BLOCK_LEN),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_compress_sse41(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| unsafe {
        benchmarks::compress_sse41(
            &mut state,
            array_ref!(input.get(), 0, BLOCK_LEN),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_fficompress_portable(b: &mut Bencher) {
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| unsafe {
        baokeshed::c::ffi::compress_portable(
            state.as_mut_ptr(),
            input.get().as_ptr(),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "c_sse41", feature = "c_native"))]
fn bench_fficompress_sse41(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| unsafe {
        baokeshed::c::ffi::compress_sse41(
            state.as_mut_ptr(),
            input.get().as_ptr(),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(feature = "c_avx512", feature = "c_native"))]
fn bench_fficompress_avx512(b: &mut Bencher) {
    if !baokeshed::c::is_avx512_detected() {
        return;
    }
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| unsafe {
        baokeshed::c::ffi::compress_avx512(
            state.as_mut_ptr(),
            input.get().as_ptr(),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
fn bench_compress64_portable(b: &mut Bencher) {
    let mut state = [0; 4];
    let mut input = RandomInput::new(b, baokeshed::portable64::BLOCK_LEN);
    b.iter(|| {
        baokeshed::portable64::compress(
            &mut state,
            array_ref!(input.get(), 0, baokeshed::portable64::BLOCK_LEN),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_fficompress64_portable(b: &mut Bencher) {
    let mut state = [0; 4];
    let mut input = RandomInput::new(b, baokeshed::portable64::BLOCK_LEN);
    b.iter(|| unsafe {
        baokeshed::c64::ffi::baokeshed64_compress_portable(
            state.as_mut_ptr(),
            input.get().as_ptr(),
            BLOCK_LEN as u8,
            0,
            0,
        );
    });
}
