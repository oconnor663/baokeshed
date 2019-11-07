#![feature(test)]

extern crate test;

use arrayref::array_ref;
use baokeshed::*;
use rand::prelude::*;
use test::Bencher;

const LONG: usize = 1 << 16; // 64 KiB

const BLOCK_LEN_32: usize = BLOCK_LEN;
const BLOCK_LEN_64: usize = portable64::BLOCK_LEN;
const CHUNK_LEN_32: usize = CHUNK_LEN;
const CHUNK_LEN_64: usize = portable64::CHUNK_LEN;

#[allow(dead_code)]
const CHUNK_OFFSET_DELTAS: [u64; 17] = [
    CHUNK_LEN as u64 * 0,
    CHUNK_LEN as u64 * 1,
    CHUNK_LEN as u64 * 2,
    CHUNK_LEN as u64 * 3,
    CHUNK_LEN as u64 * 4,
    CHUNK_LEN as u64 * 5,
    CHUNK_LEN as u64 * 6,
    CHUNK_LEN as u64 * 7,
    CHUNK_LEN as u64 * 8,
    CHUNK_LEN as u64 * 9,
    CHUNK_LEN as u64 * 10,
    CHUNK_LEN as u64 * 11,
    CHUNK_LEN as u64 * 12,
    CHUNK_LEN as u64 * 13,
    CHUNK_LEN as u64 * 14,
    CHUNK_LEN as u64 * 15,
    CHUNK_LEN as u64 * 16,
];

#[allow(dead_code)]
const PARENT_OFFSET_DELTAS: [u64; 17] = [0; 17];

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
fn bench_compress_portable_rust(b: &mut Bencher) {
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_32);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_32);
    b.iter(|| portable::compress(&mut state, input, BLOCK_LEN_32 as u8, 0, 0));
}

#[bench]
fn bench_compress_portable_rust64(b: &mut Bencher) {
    let mut state = [1; 4];
    let mut r = RandomInput::new(b, BLOCK_LEN_64);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_64);
    b.iter(|| portable64::compress(&mut state, input, BLOCK_LEN_64 as u8, 0, 0));
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_compress_portable_c(b: &mut Bencher) {
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_32);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_32);
    b.iter(|| unsafe {
        c::ffi::compress_portable(state.as_mut_ptr(), input.as_ptr(), BLOCK_LEN_32 as u8, 0, 0)
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_compress_portable_c64(b: &mut Bencher) {
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_64);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_64);
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_compress_portable(
            state.as_mut_ptr(),
            input.as_ptr(),
            BLOCK_LEN_64 as u8,
            0,
            0,
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_compress_sse41_rust(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_32);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_32);
    b.iter(|| unsafe { sse41::compress(&mut state, input, BLOCK_LEN_32 as u8, 0, 0) });
}

#[bench]
#[cfg(feature = "c_sse41")]
fn bench_compress_sse41_c(b: &mut Bencher) {
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_32);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_32);
    b.iter(|| unsafe {
        c::ffi::compress_sse41(state.as_mut_ptr(), input.as_ptr(), BLOCK_LEN_32 as u8, 0, 0)
    });
}

#[bench]
#[cfg(feature = "c_avx2")]
fn bench_compress_avx2_c64(b: &mut Bencher) {
    let mut state = [1; 4];
    let mut r = RandomInput::new(b, BLOCK_LEN_64);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_64);
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_compress_avx2(
            state.as_mut_ptr(),
            input.as_ptr(),
            BLOCK_LEN_64 as u8,
            0,
            0,
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_compress_avx512_c(b: &mut Bencher) {
    let mut state = [1; 8];
    let mut r = RandomInput::new(b, BLOCK_LEN_32);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_32);
    b.iter(|| unsafe {
        c::ffi::compress_avx512(state.as_mut_ptr(), input.as_ptr(), BLOCK_LEN_32 as u8, 0, 0)
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_compress_avx512_c64(b: &mut Bencher) {
    let mut state = [1; 4];
    let mut r = RandomInput::new(b, BLOCK_LEN_64);
    let input = array_ref!(r.get(), 0, BLOCK_LEN_64);
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_compress_avx512(
            state.as_mut_ptr(),
            input.as_ptr(),
            BLOCK_LEN_64 as u8,
            0,
            0,
        )
    });
}

#[bench]
#[cfg(feature = "c_sse41")]
fn bench_chunks_x02_sse41_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_sse41(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x02_avx512_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_avx512(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_neon")]
fn bench_chunks_x02_neon_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_neon(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_chunks_x04_sse41_rust(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        sse41::hash4(
            &inputs,
            CHUNK_LEN / BLOCK_LEN,
            &key,
            0,
            0,
            0,
            0,
            0,
            &mut out,
        )
    });
}

#[bench]
#[cfg(feature = "c_sse41")]
fn bench_chunks_x04_sse41_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_sse41(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x04_avx512_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_avx512(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_neon")]
fn bench_chunks_x04_neon_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_neon(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx2")]
fn bench_chunks_x04_avx2_c64(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash4_avx2(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x04_avx512_c64(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash4_avx512(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_chunks_x08_avx2_rust(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        avx2::hash8(
            &inputs,
            CHUNK_LEN / BLOCK_LEN,
            &key,
            0,
            0,
            0,
            0,
            0,
            &mut out,
        )
    });
}

#[bench]
#[cfg(feature = "c_avx2")]
fn bench_chunks_x08_avx2_c(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash8_avx2(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x08_avx512_c(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash8_avx512(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x08_avx512_c64(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN_64 * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs
        .iter_mut()
        .zip(input_bytes.chunks_exact(CHUNK_LEN_64))
    {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash8_avx512(
            inputs.as_ptr(),
            CHUNK_LEN_64 / BLOCK_LEN_64,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_chunks_x16_avx512_c(b: &mut Bencher) {
    const N: usize = 16;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, CHUNK_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(CHUNK_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash16_avx512(
            inputs.as_ptr(),
            CHUNK_LEN / BLOCK_LEN,
            key.as_ptr(),
            0,
            CHUNK_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_sse41")]
fn bench_parents_x02_sse41_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_sse41(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x02_avx512_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_neon")]
fn bench_parents_x02_neon_c64(b: &mut Bencher) {
    const N: usize = 2;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash2_neon(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_parents_x04_sse41_rust(b: &mut Bencher) {
    if !is_x86_feature_detected!("sse4.1") {
        return;
    }
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe { sse41::hash4(&inputs, 1, &key, 0, 0, 0, 0, 0, &mut out) });
}

#[bench]
#[cfg(feature = "c_sse41")]
fn bench_parents_x04_sse41_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_sse41(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x04_avx512_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_neon")]
fn bench_parents_x04_neon_c(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash4_neon(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx2")]
fn bench_parents_x04_avx2_c64(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash4_avx2(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x04_avx512_c64(b: &mut Bencher) {
    const N: usize = 4;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash4_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_parents_x08_avx2_rust(b: &mut Bencher) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe { avx2::hash8(&inputs, 1, &key, 0, 0, 0, 0, 0, &mut out) });
}

#[bench]
#[cfg(feature = "c_avx2")]
fn bench_parents_x08_avx2_c(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash8_avx2(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x08_avx512_c(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash8_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x08_avx512_c64(b: &mut Bencher) {
    const N: usize = 8;
    let key = [1; 4];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c64::ffi::baokeshed64_hash8_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
#[cfg(feature = "c_avx512")]
fn bench_parents_x16_avx512_c(b: &mut Bencher) {
    const N: usize = 16;
    let key = [1; 8];
    let mut out = [0; OUT_LEN * N];
    let mut r = RandomInput::new(b, 2 * OUT_LEN * N);
    let input_bytes = r.get();
    let mut inputs = [std::ptr::null(); N];
    for (input, chunk) in inputs.iter_mut().zip(input_bytes.chunks_exact(2 * OUT_LEN)) {
        *input = chunk.as_ptr();
    }
    b.iter(|| unsafe {
        c::ffi::hash16_avx512(
            inputs.as_ptr(),
            1,
            key.as_ptr(),
            0,
            PARENT_OFFSET_DELTAS.as_ptr(),
            0,
            0,
            0,
            out.as_mut_ptr(),
        )
    });
}

#[bench]
fn bench_hash_01_long_rust(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_02_chunk_rust(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_03_block_rust(b: &mut Bencher) {
    let mut r = RandomInput::new(b, BLOCK_LEN);
    let input = r.get();
    b.iter(|| baokeshed::hash(input));
}

#[bench]
fn bench_hasher_01_long_rust(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_02_chunk_rust(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_03_block_rust(b: &mut Bencher) {
    let mut r = RandomInput::new(b, BLOCK_LEN);
    let input = r.get();
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.update(input);
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_hasher_01_long_c(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_hasher_02_chunk_c(b: &mut Bencher) {
    let mut input = RandomInput::new(b, CHUNK_LEN);
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input.get());
        hasher.finalize()
    });
}

#[bench]
#[cfg(feature = "c_portable")]
fn bench_hasher_03_block_c(b: &mut Bencher) {
    let mut r = RandomInput::new(b, BLOCK_LEN);
    let input = r.get();
    b.iter(|| {
        let mut hasher = c::Hasher::new(&[0; KEY_LEN], 0);
        hasher.update(input);
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
