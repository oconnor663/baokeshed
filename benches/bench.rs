#![feature(test)]

extern crate test;

use baokeshed::*;
use rand::prelude::*;
use test::Bencher;

const SHORT: usize = baokeshed::BLOCK_BYTES;

const MEDIUM: usize = baokeshed::CHUNK_BYTES;

const LONG: usize = 1 << 20;

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
fn bench_hash_slice_short(b: &mut Bencher) {
    let mut input = RandomInput::new(b, SHORT);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_slice_medium(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MEDIUM);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hash_slice_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| baokeshed::hash(input.get()));
}

#[bench]
fn bench_hasher_short(b: &mut Bencher) {
    let mut input = RandomInput::new(b, SHORT);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.append(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_medium(b: &mut Bencher) {
    let mut input = RandomInput::new(b, MEDIUM);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.append(input.get());
        hasher.finalize()
    });
}

#[bench]
fn bench_hasher_long(b: &mut Bencher) {
    let mut input = RandomInput::new(b, LONG);
    b.iter(|| {
        let mut hasher = Hasher::new();
        hasher.append(input.get());
        hasher.finalize()
    });
}
