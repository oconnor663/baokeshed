//! FFI wrappers of the C implementation
//!
//! This module is intended mainly for benchmarking, and we also test that it
//! produces the same output.

use crate::{Word, BLOCK_LEN, KEY_LEN, OUT_LEN};
use std::mem::MaybeUninit;

// A wrapper function for unit testing and benchmarking.
pub fn compress(
    state: &mut [Word; 8],
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    offset: u64,
    internal_flags: u8,
    context: Word,
) {
    unsafe {
        ffi::compress(
            state.as_mut_ptr(),
            block.as_ptr(),
            block_len,
            offset,
            internal_flags,
            context,
        );
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ChunkState {
    state: [u32; 8usize],
    offset: u64,
    count: u16,
    buf: [u8; 64usize],
    buf_len: u8,
}

// These safe wrapper methods are just for unit testing. The real callers for
// these functions are in C.
impl ChunkState {
    pub fn new(key_words: &[Word; 8], offset: u64) -> ChunkState {
        let mut state: MaybeUninit<ChunkState> = MaybeUninit::uninit();
        unsafe {
            ffi::chunk_state_init(state.as_mut_ptr(), key_words.as_ptr(), offset);
            state.assume_init()
        }
    }

    pub fn update(&mut self, input: &[u8], context: Word) {
        unsafe {
            ffi::chunk_state_update(self, input.as_ptr(), input.len(), context);
        }
    }

    pub fn finalize(&self, context: Word, is_root: bool) -> [u8; OUT_LEN] {
        let mut out = [0; OUT_LEN];
        unsafe {
            ffi::chunk_state_finalize(self, context, is_root, out.as_mut_ptr());
        }
        out
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Hasher {
    pub chunk: ChunkState,
    pub key_words: [u32; 8usize],
    pub context: u32,
    pub subtree_hashes_len: u8,
    pub subtree_hashes: [u8; 1664usize],
}

impl Hasher {
    pub fn new(key: &[u8; KEY_LEN], context: u32) -> Hasher {
        let mut hasher: MaybeUninit<Hasher> = MaybeUninit::uninit();
        unsafe {
            ffi::hasher_init(hasher.as_mut_ptr(), key.as_ptr(), context);
            hasher.assume_init()
        }
    }

    pub fn update(&mut self, input: &[u8]) {
        unsafe {
            ffi::hasher_update(self, input.as_ptr() as _, input.len());
        }
    }

    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let mut out = [0; OUT_LEN];
        unsafe {
            ffi::hasher_finalize(self, out.as_mut_ptr());
        }
        out
    }
}

mod ffi {
    use super::ChunkState;
    use super::Hasher;

    extern "C" {
        pub fn compress(
            state: *mut u32,
            block: *const u8,
            block_len: u8,
            offset: u64,
            internal_flags: u8,
            context: u32,
        );
        pub fn chunk_state_init(state: *mut ChunkState, key_words: *const u32, offset: u64);
        pub fn chunk_state_update(
            state: *mut ChunkState,
            input: *const u8,
            input_len: usize,
            context: u32,
        );
        pub fn chunk_state_finalize(
            state: *const ChunkState,
            context: u32,
            is_root: bool,
            out: *mut u8,
        );
        pub fn hasher_init(hasher: *mut Hasher, key: *const u8, context: u32);
        pub fn hasher_update(
            hasher: *mut Hasher,
            input: *const ::std::os::raw::c_void,
            input_len: usize,
        );
        pub fn hasher_finalize(hasher: *const Hasher, out: *mut u8);
    }
}

#[cfg(test)]
mod test {
    use crate::{BLOCK_LEN, CHUNK_LEN, KEY_LEN, WORD_BITS};

    #[test]
    fn test_compare_compress() {
        let initial_state = [1, 2, 3, 4, 5, 6, 7, 8];
        let block_len: u8 = 27;
        let mut block = [0; BLOCK_LEN];
        crate::test::paint_test_input(&mut block[..block_len as usize]);
        // Use an offset with set bits in both 32-bit words.
        let offset = ((5 * CHUNK_LEN as u64) << WORD_BITS) + 6 * CHUNK_LEN as u64;
        let flags = crate::Flags::CHUNK_END | crate::Flags::ROOT;
        let context = 23;

        let mut rust_state = initial_state;
        crate::portable::compress(
            &mut rust_state,
            &block,
            block_len,
            offset as u64,
            flags.bits(),
            context,
        );

        let mut c_state = initial_state;
        super::compress(
            &mut c_state,
            &block,
            block_len,
            offset as u64,
            flags.bits(),
            context,
        );

        assert_eq!(rust_state, c_state);
    }

    #[test]
    fn test_compare_hash_chunk() {
        const CASES: &[usize] = &[
            0,
            1,
            BLOCK_LEN - 1,
            BLOCK_LEN,
            BLOCK_LEN + 1,
            CHUNK_LEN - 1,
            CHUNK_LEN,
        ];
        let mut input_buf = [0; CHUNK_LEN];
        crate::test::paint_test_input(&mut input_buf);
        let key_words = [10, 11, 12, 13, 14, 15, 16, 17];
        let offset = 0;
        let context = 100;
        for &is_root in &[crate::IsRoot::NotRoot, crate::IsRoot::Root] {
            dbg!(is_root);
            for &case in CASES {
                dbg!(case);
                let input = &input_buf[..case];

                let mut rust_chunk = crate::ChunkState::new(&key_words, offset);
                let platform = crate::platform::Platform::detect();
                rust_chunk.update(input, context, platform);
                let rust_out = rust_chunk.finalize(is_root, context, platform).read();

                // First test at once.
                let is_root_bool = is_root.flag().bits() > 0;
                let mut c_chunk = super::ChunkState::new(&key_words, offset);
                c_chunk.update(input, context);
                let c_out = c_chunk.finalize(context, is_root_bool);
                assert_eq!(rust_out, c_out);

                // Then test one byte at a time.
                let is_root_bool = is_root.flag().bits() > 0;
                let mut c_chunk = super::ChunkState::new(&key_words, offset);
                for &byte in input {
                    c_chunk.update(&[byte], context);
                }
                let c_out = c_chunk.finalize(context, is_root_bool);
                assert_eq!(rust_out, c_out);
            }
        }
    }

    #[test]
    fn test_compare_hash_tree() {
        let mut input_buf = [0; crate::test::TEST_CASES_MAX];
        crate::test::paint_test_input(&mut input_buf);
        let key = [5; KEY_LEN];
        let context = 999;

        for &case in crate::test::TEST_CASES {
            dbg!(case);
            let input = &input_buf[..case];

            let rust_hash = crate::hash_keyed_contextified_xof(input, &key, context).read();

            // First test at once.
            let mut c_hasher = super::Hasher::new(&key, context);
            c_hasher.update(input);
            let c_hash = c_hasher.finalize();
            assert_eq!(rust_hash, c_hash);

            // Then test one byte at a time.
            let mut c_hasher = super::Hasher::new(&key, context);
            for &byte in input {
                c_hasher.update(&[byte]);
            }
            let c_hash = c_hasher.finalize();
            assert_eq!(rust_hash, c_hash);
        }
    }
}
