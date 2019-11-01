//! FFI wrappers of the C implementation
//!
//! This module is intended mainly for benchmarking, and we also test that it
//! produces the same output.

use crate::{KEY_LEN, OUT_LEN};
use std::mem::MaybeUninit;

// Methods on this chunk state are only used for testing. They're defined in
// the `test` module below.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ChunkState {
    state: [u32; 8usize],
    key: [u32; 8usize],
    offset: u64,
    count: u16,
    buf: [u8; 64usize],
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

mod ffi {
    extern "C" {
        pub fn baokeshed64_hasher_init(self_: *mut super::Hasher, key: *const u8, flags: u8);
        pub fn baokeshed64_hasher_update(
            self_: *mut super::Hasher,
            input: *const ::std::os::raw::c_void,
            input_len: usize,
        );
        pub fn baokeshed64_hasher_finalize(self_: *const super::Hasher, out: *mut u8);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{bytes_from_state_words, Flags, Word, BLOCK_LEN, CHUNK_LEN, WORD_BITS};
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

    #[cfg(any(feature = "c_avx512", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn is_avx512_detected() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
    }

    // FFI functions that we only call in tests.
    mod ffi {
        extern "C" {
            pub fn baokeshed64_compress_portable(
                state: *mut u32,
                block: *const u8,
                block_len: u8,
                offset: u64,
                flags: u8,
            );
            pub fn baokeshed64_hash_many_portable(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                flags: u8,
                flags_start: u8,
                flags_end: u8,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_sse41", feature = "c_native"))]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            pub fn hash_many_sse41(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                flags: u8,
                flags_start: u8,
                flags_end: u8,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_avx2", feature = "c_native"))]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            pub fn hash_many_avx2(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                flags: u8,
                flags_start: u8,
                flags_end: u8,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_avx512", feature = "c_native"))]
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            pub fn hash_many_avx512(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                flags: u8,
                flags_start: u8,
                flags_end: u8,
                out: *mut u8,
            );
            #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
            #[cfg(feature = "c_armv7neon")]
            pub fn hash_many_neon(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                flags: u8,
                flags_start: u8,
                flags_end: u8,
                out: *mut u8,
            );
            pub fn chunk_state_init(self_: *mut super::ChunkState, key: *const u32, flags: u8);
            pub fn chunk_state_update(
                self_: *mut super::ChunkState,
                input: *const u8,
                input_len: usize,
            );
            pub fn chunk_state_finalize(
                self_: *const super::ChunkState,
                is_root: bool,
                out: *mut u8,
            );
        }
    }

    impl ChunkState {
        pub fn new(key_words: &[Word; 8], flags: u8) -> ChunkState {
            let mut state: MaybeUninit<ChunkState> = MaybeUninit::uninit();
            unsafe {
                ffi::chunk_state_init(state.as_mut_ptr(), key_words.as_ptr(), flags);
                state.assume_init()
            }
        }

        pub fn update(&mut self, input: &[u8]) {
            unsafe {
                ffi::chunk_state_update(self, input.as_ptr(), input.len());
            }
        }

        pub fn finalize(&self, is_root: bool) -> [u8; OUT_LEN] {
            let mut out = [0; OUT_LEN];
            unsafe {
                ffi::chunk_state_finalize(self, is_root, out.as_mut_ptr());
            }
            out
        }
    }

    type CompressFn = unsafe extern "C" fn(
        state: *mut u32,
        block: *const u8,
        block_len: u8,
        offset: u64,
        flags: u8,
    );

    fn compare_compress_fn(compress_fn: CompressFn) {
        let initial_state = [1, 2, 3, 4, 5, 6, 7, 8];
        let block_len: u8 = 27;
        let mut block = [0; BLOCK_LEN];
        crate::test::paint_test_input(&mut block[..block_len as usize]);
        // Use an offset with set bits in both 32-bit words.
        let offset = ((5 * CHUNK_LEN as u64) << WORD_BITS) + 6 * CHUNK_LEN as u64;
        let flags = Flags::CHUNK_END | Flags::ROOT;

        let mut rust_state = initial_state;
        crate::portable::compress(
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
        compare_compress_fn(ffi::compress_portable);
    }

    #[test]
    #[cfg(any(feature = "c_sse41", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress_sse41() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        compare_compress_fn(ffi::compress_sse41);
    }

    #[test]
    #[cfg(any(feature = "c_avx512", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_compress_avx512() {
        if !is_avx512_detected() {
            return;
        }
        compare_compress_fn(ffi::compress_avx512);
    }

    type HashManyFn = unsafe extern "C" fn(
        inputs: *const *const u8,
        num_inputs: usize,
        blocks: usize,
        key_words: *const u32,
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
        let key_words = [21, 22, 23, 24, 25, 26, 27, 28];
        let offset = 99 * CHUNK_LEN as u64;

        // First hash chunks.
        let mut chunks = Vec::new();
        for i in 0..NUM_INPUTS {
            chunks.push(array_ref!(input_buf, i * CHUNK_LEN, CHUNK_LEN));
        }
        let platform = crate::platform::Platform::detect();
        let mut rust_out = [0; NUM_INPUTS * OUT_LEN];
        platform.hash_many(
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
        let platform = crate::platform::Platform::detect();
        let mut rust_out = [0; NUM_INPUTS * OUT_LEN];
        platform.hash_many(
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
        compare_hash_many_fn(ffi::hash_many_portable);
    }

    #[test]
    #[cfg(any(feature = "c_sse41", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_hash_many_sse41() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        compare_hash_many_fn(ffi::hash_many_sse41);
    }

    #[test]
    #[cfg(any(feature = "c_avx2", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_hash_many_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        compare_hash_many_fn(ffi::hash_many_avx2);
    }

    #[test]
    #[cfg(any(feature = "c_avx512", feature = "c_native"))]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_hash_many_avx512() {
        if !is_avx512_detected() {
            return;
        }
        compare_hash_many_fn(ffi::hash_many_avx512);
    }

    #[test]
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[cfg(feature = "c_armv7neon")]
    fn test_hash_many_neon() {
        compare_hash_many_fn(ffi::hash_many_neon);
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
        let flags = Flags::KEYED_HASH;
        for &is_root in &[crate::IsRoot::NotRoot, crate::IsRoot::Root] {
            dbg!(is_root);
            for &case in CASES {
                dbg!(case);
                let input = &input_buf[..case];

                let mut rust_chunk = crate::ChunkState::new(&key_words, flags);
                rust_chunk.update(input);
                let rust_out = rust_chunk.finalize(is_root).to_hash();

                // First test at once.
                let is_root_bool = is_root.flag().bits() > 0;
                let mut c_chunk = super::ChunkState::new(&key_words, flags.bits());
                c_chunk.update(input);
                let c_out = c_chunk.finalize(is_root_bool);
                assert_eq!(rust_out, c_out);

                // Then test one byte at a time.
                let is_root_bool = is_root.flag().bits() > 0;
                let mut c_chunk = super::ChunkState::new(&key_words, flags.bits());
                for &byte in input {
                    c_chunk.update(&[byte]);
                }
                let c_out = c_chunk.finalize(is_root_bool);
                assert_eq!(rust_out, c_out);
            }
        }
    }

    #[test]
    fn test_compare_hash_tree() {
        let mut input_buf = [0; crate::test::TEST_CASES_MAX];
        crate::test::paint_test_input(&mut input_buf);
        let key_words = [5; 8];
        let key_bytes = bytes_from_state_words(&key_words);

        for &case in crate::test::TEST_CASES {
            dbg!(case);
            let input = &input_buf[..case];

            let rust_hash = crate::hash_internal(input, &key_words, Flags::KEYED_HASH).to_hash();

            // First test at once.
            let mut c_hasher = super::Hasher::new(&key_bytes, Flags::KEYED_HASH.bits());
            c_hasher.update(input);
            let c_hash = c_hasher.finalize();
            assert_eq!(rust_hash, c_hash);

            // Then test one byte at a time.
            let mut c_hasher = super::Hasher::new(&key_bytes, Flags::KEYED_HASH.bits());
            for &byte in input {
                c_hasher.update(&[byte]);
            }
            let c_hash = c_hasher.finalize();
            assert_eq!(rust_hash, c_hash);
        }
    }
}
