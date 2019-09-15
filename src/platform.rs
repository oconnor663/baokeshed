use crate::{
    avx2, bytes_from_state_words, iv, portable, sse41, Word, BLOCK_LEN, CHUNK_LEN, OUT_LEN,
};
use arrayref::{array_mut_ref, array_ref};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const MAX_SIMD_DEGREE: usize = 8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub const MAX_SIMD_DEGREE: usize = 1;

type CompressionFn = unsafe fn(
    state: &mut [Word; 8],
    block: &[u8; BLOCK_LEN],
    block_len: Word,
    offset: u64,
    flags: Word,
);

type HashManyFn = unsafe fn(
    inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    offset: u64,
    offset_delta: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    out: &mut [u8],
);

#[derive(Clone, Copy)]
pub struct Platform {
    compression_fn: CompressionFn,
    hash_many_fn: HashManyFn,
    simd_degree: usize,
}

impl Platform {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return Self {
                    compression_fn: sse41::compress, // no AVX2 implementation
                    hash_many_fn: hash_many_avx2,
                    simd_degree: avx2::DEGREE,
                };
            }
            if is_x86_feature_detected!("sse4.1") {
                return Self {
                    compression_fn: sse41::compress,
                    hash_many_fn: hash_many_sse41,
                    simd_degree: sse41::DEGREE,
                };
            }
        }
        Self {
            compression_fn: portable::compress,
            hash_many_fn: hash_many_portable,
            simd_degree: 1,
        }
    }

    pub fn simd_degree(&self) -> usize {
        self.simd_degree
    }

    pub fn compress(
        &self,
        state: &mut [Word; 8],
        block: &[u8; BLOCK_LEN],
        block_len: Word,
        offset: u64,
        flags: Word,
    ) {
        // Safe because detect() checked for platform support.
        unsafe {
            (self.compression_fn)(state, block, block_len, offset, flags);
        }
    }

    pub fn hash_many_chunks(
        &self,
        chunks: &[&[u8; CHUNK_LEN]],
        key: &[Word; 8],
        offset: u64,
        flags_all: Word,
        flags_start: Word,
        flags_end: Word,
        out: &mut [u8],
    ) {
        let blocks = CHUNK_LEN / BLOCK_LEN;
        unsafe {
            // Safe because the layout of arrays is guaranteed, and because the
            // `blocks` count is determined statically from the argument type.
            let chunk_ptrs: &[*const u8] =
                core::slice::from_raw_parts(chunks.as_ptr() as *const *const u8, chunks.len());
            // Safe because detect() checked for platform support.
            (self.hash_many_fn)(
                chunk_ptrs,
                blocks,
                key,
                offset,
                CHUNK_LEN as u64,
                flags_all,
                flags_start,
                flags_end,
                out,
            );
        }
    }

    pub fn hash_many_parents(
        &self,
        parents: &[&[u8; BLOCK_LEN]],
        key: &[Word; 8],
        flags: Word,
        out: &mut [u8],
    ) {
        let blocks = 1;
        let offset = 0;
        let offset_delta = 0;
        let flags_start = 0;
        let flags_end = 0;
        unsafe {
            // Safe because the layout of arrays is guaranteed, and because the
            // `blocks` count is determined statically from the argument type.
            let parent_ptrs: &[*const u8] =
                core::slice::from_raw_parts(parents.as_ptr() as *const *const u8, parents.len());
            // Safe because detect() checked for platform support.
            (self.hash_many_fn)(
                parent_ptrs,
                blocks,
                key,
                offset,
                offset_delta,
                flags,
                flags_start,
                flags_end,
                out,
            );
        }
    }
}

unsafe fn hash_many_serial(
    mut inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    mut offset: u64,
    offset_delta: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
    compression_fn: CompressionFn,
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    while inputs.len() >= 1 && out.len() >= OUT_LEN {
        let mut state = iv(key);
        let input_blocks = inputs[0] as *const [u8; BLOCK_LEN];
        let mut flags = flags_all | flags_start;
        for block in 0..blocks {
            if block + 1 == blocks {
                flags |= flags_end;
            }
            compression_fn(
                &mut state,
                &*input_blocks.add(block),
                BLOCK_LEN as Word,
                offset,
                flags,
            );
            flags = flags_all;
        }
        *array_mut_ref!(out, 0, OUT_LEN) = bytes_from_state_words(&state);
        inputs = &inputs[1..];
        offset += offset_delta;
        out = &mut out[OUT_LEN..];
    }
}

unsafe fn hash_many_portable(
    inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    offset: u64,
    offset_delta: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    out: &mut [u8],
) {
    hash_many_serial(
        inputs,
        blocks,
        key,
        offset,
        offset_delta,
        flags_all,
        flags_start,
        flags_end,
        out,
        portable::compress,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hash_many_sse41(
    mut inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    mut offset: u64,
    offset_delta: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    while inputs.len() >= sse41::DEGREE && out.len() >= sse41::DEGREE * OUT_LEN {
        sse41::compress4_loop(
            array_ref!(inputs, 0, sse41::DEGREE),
            blocks,
            key,
            offset,
            offset_delta,
            flags_all,
            flags_start,
            flags_end,
            array_mut_ref!(out, 0, sse41::DEGREE * OUT_LEN),
        );
        inputs = &inputs[sse41::DEGREE..];
        offset += sse41::DEGREE as u64 * offset_delta;
        out = &mut out[sse41::DEGREE * OUT_LEN..];
    }

    hash_many_serial(
        inputs,
        blocks,
        key,
        offset,
        offset_delta,
        flags_all,
        flags_start,
        flags_end,
        out,
        sse41::compress,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hash_many_avx2(
    mut inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    mut offset: u64,
    offset_delta: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    while inputs.len() >= avx2::DEGREE && out.len() >= avx2::DEGREE * OUT_LEN {
        avx2::compress8_loop(
            array_ref!(inputs, 0, avx2::DEGREE),
            blocks,
            key,
            offset,
            offset_delta,
            flags_all,
            flags_start,
            flags_end,
            array_mut_ref!(out, 0, avx2::DEGREE * OUT_LEN),
        );
        inputs = &inputs[avx2::DEGREE..];
        offset += avx2::DEGREE as u64 * offset_delta;
        out = &mut out[avx2::DEGREE * OUT_LEN..];
    }

    hash_many_sse41(
        inputs,
        blocks,
        key,
        offset,
        offset_delta,
        flags_all,
        flags_start,
        flags_end,
        out,
    );
}

#[cfg(test)]
mod test {
    use super::*;
    use arrayvec::ArrayVec;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_hash_many_sse41() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }

        // An uneven number of inputs, to trigger all the different loops.
        const NUM_INPUTS: usize = 2 * sse41::DEGREE - 1;

        let mut input_buf = [0; NUM_INPUTS * BLOCK_LEN];
        crate::test::paint_test_input(&mut input_buf);
        let mut blocks_array = ArrayVec::<[*const u8; NUM_INPUTS]>::new();
        for i in 0..NUM_INPUTS {
            blocks_array.push(input_buf[i * BLOCK_LEN..].as_ptr());
        }

        let mut out_portable = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_portable(
                &blocks_array,
                1,
                &[42; 8],
                1001,
                1002,
                1,
                2,
                4,
                &mut out_portable,
            );
        }

        let mut out_sse41 = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_sse41(
                &blocks_array,
                1,
                &[42; 8],
                1001,
                1002,
                1,
                2,
                4,
                &mut out_sse41,
            );
        }

        assert_eq!(&out_portable[..], &out_sse41[..]);
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_hash_many_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // An uneven number of inputs, to trigger all the different loops.
        const NUM_INPUTS: usize = 2 * avx2::DEGREE - 1;

        let mut input_buf = [0; NUM_INPUTS * BLOCK_LEN];
        crate::test::paint_test_input(&mut input_buf);
        let mut blocks_array = ArrayVec::<[*const u8; NUM_INPUTS]>::new();
        for i in 0..NUM_INPUTS {
            blocks_array.push(input_buf[i * BLOCK_LEN..].as_ptr());
        }

        let mut out_portable = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_portable(
                &blocks_array,
                1,
                &[42; 8],
                1001,
                1002,
                1,
                2,
                4,
                &mut out_portable,
            );
        }

        let mut out_avx2 = [0; NUM_INPUTS * OUT_LEN];
        unsafe {
            hash_many_avx2(
                &blocks_array,
                1,
                &[42; 8],
                1001,
                1002,
                1,
                2,
                4,
                &mut out_avx2,
            );
        }

        assert_eq!(&out_portable[..], &out_avx2[..]);
    }
}
