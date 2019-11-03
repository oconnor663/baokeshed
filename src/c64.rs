//! FFI wrappers of the C implementation
//!
//! This module is intended mainly for benchmarking, and we also test that it
//! produces the same output.

use crate::{KEY_LEN, OUT_LEN};
use std::mem::MaybeUninit;

pub const BLOCK_LEN: usize = 128;
pub const WORD_BITS: usize = 64;

// Methods on this chunk state are only used for testing. They're defined in
// the `test` module below.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ChunkState {
    state: [u64; 4usize],
    key: [u64; 4usize],
    offset: u64,
    count: u16,
    buf: [u8; 128usize],
    buf_len: u8,
    flags: u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Hasher {
    chunk: ChunkState,
    subtree_hashes_len: u8,
    subtree_hashes: [u8; 1664usize],
}

impl Hasher {
    pub fn new(key: &[u8; KEY_LEN], flags: u8) -> Hasher {
        let mut hasher: MaybeUninit<Hasher> = MaybeUninit::uninit();
        unsafe {
            ffi::baokeshed64_hasher_init(hasher.as_mut_ptr(), key.as_ptr(), flags);
            hasher.assume_init()
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        unsafe {
            ffi::baokeshed64_hasher_update(self, input.as_ptr() as _, input.len());
        }
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let mut out = [0; OUT_LEN];
        unsafe {
            ffi::baokeshed64_hasher_finalize(self, out.as_mut_ptr());
        }
        out
    }
}

pub mod ffi {
    extern "C" {
        pub fn baokeshed64_hasher_init(self_: *mut super::Hasher, key: *const u8, flags: u8);
        pub fn baokeshed64_hasher_update(
            self_: *mut super::Hasher,
            input: *const ::std::os::raw::c_void,
            input_len: usize,
        );
        pub fn baokeshed64_hasher_finalize(self_: *const super::Hasher, out: *mut u8);
        pub fn baokeshed64_compress_portable(
            state: *mut u64,
            block: *const u8,
            block_len: u8,
            offset: u64,
            flags: u8,
        );
        pub fn baokeshed64_hash_many_portable(
            inputs: *const *const u8,
            num_inputs: usize,
            blocks: usize,
            key_words: *const u64,
            offset: u64,
            offset_deltas: *const u64,
            flags: u8,
            flags_start: u8,
            flags_end: u8,
            out: *mut u8,
        );
        pub fn baokeshed64_chunk_state_init(
            self_: *mut super::ChunkState,
            key: *const u64,
            flags: u8,
        );
        pub fn baokeshed64_chunk_state_update(
            self_: *mut super::ChunkState,
            input: *const u8,
            input_len: usize,
        );
        pub fn baokeshed64_chunk_state_finalize(
            self_: *const super::ChunkState,
            is_root: bool,
            out: *mut u8,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Flags, CHUNK_LEN};
    use arrayref::array_ref;

    const CHUNK_OFFSET_DELTAS: &[u64; 17] = &[
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

    const PARENT_OFFSET_DELTAS: &[u64; 17] = &[0; 17];

    impl ChunkState {
        pub fn new(key_words: &[u64; 4], flags: u8) -> ChunkState {
            let mut state: MaybeUninit<ChunkState> = MaybeUninit::uninit();
            unsafe {
                ffi::baokeshed64_chunk_state_init(state.as_mut_ptr(), key_words.as_ptr(), flags);
                state.assume_init()
            }
        }

        pub fn update(&mut self, input: &[u8]) {
            unsafe {
                ffi::baokeshed64_chunk_state_update(self, input.as_ptr(), input.len());
            }
        }

        pub fn finalize(&self, is_root: bool) -> [u8; OUT_LEN] {
            let mut out = [0; OUT_LEN];
            unsafe {
                ffi::baokeshed64_chunk_state_finalize(self, is_root, out.as_mut_ptr());
            }
            out
        }
    }

    type CompressFn = unsafe extern "C" fn(
        state: *mut u64,
        block: *const u8,
        block_len: u8,
        offset: u64,
        flags: u8,
    );

    fn compare_compress_fn(compress_fn: CompressFn) {
        let initial_state = [1, 2, 3, 4];
        let block_len: u8 = 27;
        let mut block = [0; BLOCK_LEN];
        crate::test::paint_test_input(&mut block[..block_len as usize]);
        // Use an offset with set bits in both 32-bit words.
        let offset = 5;
        let flags = Flags::CHUNK_END | Flags::ROOT;

        let mut rust_state = initial_state;
        crate::portable64::compress(
            &mut rust_state,
            &block,
            block_len,
            offset as u64,
            flags.bits(),
        );

        let mut c_state = initial_state;
        unsafe {
            compress_fn(
                c_state.as_mut_ptr(),
                block.as_ptr(),
                block_len,
                offset as u64,
                flags.bits(),
            );
        }

        assert_eq!(rust_state, c_state);
    }

    #[test]
    fn test_compress_portable() {
        compare_compress_fn(ffi::baokeshed64_compress_portable);
    }

    type HashManyFn = unsafe extern "C" fn(
        inputs: *const *const u8,
        num_inputs: usize,
        blocks: usize,
        key_words: *const u64,
        offset: u64,
        offset_deltas: *const u64,
        flags: u8,
        flags_start: u8,
        flags_end: u8,
        out: *mut u8,
    );

    fn compare_hash_many_fn(hash_many_fn: HashManyFn) {
        // 31 (16 + 8 + 4 + 2 + 1) inputs
        const NUM_INPUTS: usize = 31;
        let mut input_buf = [0; CHUNK_LEN * NUM_INPUTS];
        crate::test::paint_test_input(&mut input_buf);
        let key_words = [21, 22, 23, 24];
        let offset = 99 * CHUNK_LEN as u64;

        // First hash chunks.
        let mut chunks = Vec::new();
        for i in 0..NUM_INPUTS {
            chunks.push(array_ref!(input_buf, i * CHUNK_LEN, CHUNK_LEN));
        }
        let mut rust_out = [0; NUM_INPUTS * OUT_LEN];
        crate::portable64::hash_many(
            &chunks,
            &key_words,
            offset,
            CHUNK_LEN as u64,
            Flags::KEYED_HASH.bits(),
            Flags::CHUNK_START.bits(),
            Flags::CHUNK_END.bits(),
            &mut rust_out,
        );

        let mut c_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_fn(
                chunks.as_ptr() as _,
                NUM_INPUTS,
                CHUNK_LEN / BLOCK_LEN,
                key_words.as_ptr(),
                offset,
                CHUNK_OFFSET_DELTAS.as_ptr(),
                Flags::KEYED_HASH.bits(),
                Flags::CHUNK_START.bits(),
                Flags::CHUNK_END.bits(),
                c_out.as_mut_ptr(),
            );
        }
        for n in 0..NUM_INPUTS {
            dbg!(n);
            assert_eq!(
                &rust_out[n * OUT_LEN..][..OUT_LEN],
                &c_out[n * OUT_LEN..][..OUT_LEN]
            );
        }

        // Then hash parents.
        let mut parents = Vec::new();
        for i in 0..NUM_INPUTS {
            parents.push(array_ref!(input_buf, i * BLOCK_LEN, BLOCK_LEN));
        }
        let mut rust_out = [0; NUM_INPUTS * OUT_LEN];
        crate::portable64::hash_many(
            &parents,
            &key_words,
            0, // Parents have no offset.
            0, // Parents have no offset delta.
            Flags::PARENT.bits(),
            0, // Parents have no start flags.
            0, // Parents have no end flags.
            &mut rust_out,
        );

        let mut c_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_fn(
                parents.as_ptr() as _,
                NUM_INPUTS,
                1,
                key_words.as_ptr(),
                0,
                PARENT_OFFSET_DELTAS.as_ptr(),
                Flags::PARENT.bits(),
                0,
                0,
                c_out.as_mut_ptr(),
            );
        }
        for n in 0..NUM_INPUTS {
            dbg!(n);
            assert_eq!(
                &rust_out[n * OUT_LEN..][..OUT_LEN],
                &c_out[n * OUT_LEN..][..OUT_LEN]
            );
        }
    }

    #[test]
    fn test_hash_many_portable() {
        compare_hash_many_fn(ffi::baokeshed64_hash_many_portable);
    }
}
