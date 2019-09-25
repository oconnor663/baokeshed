#![feature(test)]

extern crate test;

use baokeshed::*;
use rand::prelude::*;
use test::Bencher;

const MEDIUM: usize = MAX_SIMD_DEGREE * CHUNK_LEN;

const LONG: usize = 1 << 24; // 16 MiB

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
#[cfg(feature = "c")]
fn bench_ffihasher_01_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = c::Hasher::new(DEFAULT_KEY, DEFAULT_CONTEXT);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c")]
fn bench_ffihasher_02_medium(b: &mut Bencher) {
    // The C code supports AVX512, so use a larger input size for this
    // benchmark when AVX512 is available.
    let mut len: usize = MEDIUM;
    #[cfg(any(feature = "c_avx512", feature = "c_native"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            len = std::cmp::max(len, 16 * CHUNK_LEN);
        }
    }
    let mut input = RandomInput::new(b, len);
    b.iter(|| {
        let mut hasher = c::Hasher::new(DEFAULT_KEY, DEFAULT_CONTEXT);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c")]
fn bench_ffihasher_03_chunk(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(DEFAULT_KEY, DEFAULT_CONTEXT);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c")]
fn bench_ffihasher_04_block(b: &mut Bencher) {
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(DEFAULT_KEY, DEFAULT_CONTEXT);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_xof(b: &mut Bencher) {
    b.bytes = OUT_LEN as u64;
    let hasher = Hasher::new();
    let mut xof = hasher.finalize_xof();
    b.iter(|| xof.read());
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_compress(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    let mut state = [0; 8];
    let mut input = RandomInput::new(b, BLOCK_LEN);
    b.iter(|| unsafe {
        let block_ptr = input.get().as_ptr() as *const [u8; BLOCK_LEN];
        compress_sse41(&mut state, &*block_ptr, BLOCK_LEN as u8, 0, 0, 0);
        state
    });
}
