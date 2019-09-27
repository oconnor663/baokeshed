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
    offset: u64,
    count: u16,
    buf: [u8; 64usize],
    buf_len: u8,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Hasher {
    chunk: ChunkState,
    key_words: [u32; 8usize],
    context: u32,
    subtree_hashes_len: u8,
    subtree_hashes: [u8; 1664usize],
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn is_avx512_detected() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
}

mod ffi {
    extern "C" {
        pub fn hasher_init(hasher: *mut super::Hasher, key: *const u8, context: u32);
        pub fn hasher_update(
            hasher: *mut super::Hasher,
            input: *const ::std::os::raw::c_void,
            input_len: usize,
        );
        pub fn hasher_finalize(hasher: *const super::Hasher, out: *mut u8);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Word, BLOCK_LEN, CHUNK_LEN, WORD_BITS};
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

    // FFI functions that we only call in tests.
    mod ffi {
        extern "C" {
            pub fn compress_portable(
                state: *mut u32,
                block: *const u8,
                block_len: u8,
                offset: u64,
                internal_flags: u8,
                context: u32,
            );
            #[cfg(any(feature = "c_sse41", feature = "c_native"))]
            pub fn compress_sse41(
                state: *mut u32,
                block: *const u8,
                block_len: u8,
                offset: u64,
                internal_flags: u8,
                context: u32,
            );
            #[cfg(any(feature = "c_avx512", feature = "c_native"))]
            pub fn compress_avx512(
                state: *mut u32,
                block: *const u8,
                block_len: u8,
                offset: u64,
                internal_flags: u8,
                context: u32,
            );
            pub fn hash_many_portable(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                internal_flags_start: u8,
                internal_flags_end: u8,
                context: u32,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_sse41", feature = "c_native"))]
            pub fn hash_many_sse41(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                internal_flags_start: u8,
                internal_flags_end: u8,
                context: u32,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_avx2", feature = "c_native"))]
            pub fn hash_many_avx2(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                internal_flags_start: u8,
                internal_flags_end: u8,
                context: u32,
                out: *mut u8,
            );
            #[cfg(any(feature = "c_avx512", feature = "c_native"))]
            pub fn hash_many_avx512(
                inputs: *const *const u8,
                num_inputs: usize,
                blocks: usize,
                key_words: *const u32,
                offset: u64,
                offset_deltas: *const u64,
                internal_flags_start: u8,
                internal_flags_end: u8,
                context: u32,
                out: *mut u8,
            );
            pub fn chunk_state_init(
                state: *mut super::ChunkState,
                key_words: *const u32,
                offset: u64,
            );
            pub fn chunk_state_update(
                state: *mut super::ChunkState,
                input: *const u8,
                input_len: usize,
                context: u32,
            );
            pub fn chunk_state_finalize(
                state: *const super::ChunkState,
                context: u32,
                is_root: bool,
                out: *mut u8,
            );
        }
    }

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

    type CompressFn = unsafe extern "C" fn(
        state: *mut u32,
        block: *const u8,
        block_len: u8,
        offset: u64,
        internal_flags: u8,
        context: u32,
    );

    fn compare_compress_fn(compress_fn: CompressFn) {
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
        unsafe {
            compress_fn(
                c_state.as_mut_ptr(),
                block.as_ptr(),
                block_len,
                offset as u64,
                flags.bits(),
                context,
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
        internal_flags_start: u8,
        internal_flags_end: u8,
        context: u32,
        out: *mut u8,
    );

    fn compare_hash_many_fn(hash_many_fn: HashManyFn) {
        // 31 (16 + 8 + 4 + 2 + 1) inputs
        const NUM_INPUTS: usize = 31;
        let mut input_buf = [0; CHUNK_LEN * NUM_INPUTS];
        crate::test::paint_test_input(&mut input_buf);
        let key_words = [21, 22, 23, 24, 25, 26, 27, 28];
        let offset = 99 * CHUNK_LEN as u64;
        let context = 255;

        // First hash chunks.
        let mut chunks = Vec::new();
        for i in 0..NUM_INPUTS {
            chunks.push(array_ref!(input_buf, i * CHUNK_LEN, CHUNK_LEN));
        }
        let platform = crate::platform::Platform::detect();
        let mut rust_out = [0; NUM_INPUTS * OUT_LEN];
        platform.hash_many_chunks(&chunks, &key_words, offset, context, &mut rust_out);

        let mut c_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_fn(
                chunks.as_ptr() as _,
                NUM_INPUTS,
                CHUNK_LEN / BLOCK_LEN,
                key_words.as_ptr(),
                offset,
                CHUNK_OFFSET_DELTAS.as_ptr(),
                crate::Flags::CHUNK_START.bits(),
                crate::Flags::CHUNK_END.bits(),
                context,
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
        platform.hash_many_parents(&parents, &key_words, context, &mut rust_out);

        let mut c_out = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_fn(
                parents.as_ptr() as _,
                NUM_INPUTS,
                1,
                key_words.as_ptr(),
                0,
                PARENT_OFFSET_DELTAS.as_ptr(),
                crate::Flags::PARENT.bits(),
                0,
                context,
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
