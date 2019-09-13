use crate::{
    avx2, bytes_from_state_words, iv, portable, sse41, Word, BLOCK_BYTES, CHUNK_BYTES, OUT_BYTES,
};
use arrayref::{array_mut_ref, array_ref};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const SIMD_DEGREE: usize = 8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub const SIMD_DEGREE: usize = 1;

type CompressionFn = unsafe fn(
    state: &mut [Word; 8],
    block: &[u8; BLOCK_BYTES],
    block_len: Word,
    offset: u64,
    flags: Word,
);

type HashManyFn = unsafe fn(
    inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    offsets: &[u64],
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    out: &mut [u8],
);

pub struct Platform {
    compression_fn: CompressionFn,
    hash_many_fn: HashManyFn,
}

impl Platform {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return Self {
                    compression_fn: sse41::compress, // no AVX2 implementation
                    hash_many_fn: hash_many_avx2,
                };
            }
            if is_x86_feature_detected!("sse4.1") {
                return Self {
                    compression_fn: sse41::compress,
                    hash_many_fn: hash_many_sse41,
                };
            }
        }
        Self {
            compression_fn: portable::compress,
            hash_many_fn: hash_many_portable,
        }
    }

    pub fn compress(
        &self,
        state: &mut [Word; 8],
        block: &[u8; BLOCK_BYTES],
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
        chunks: &[&[u8; CHUNK_BYTES]],
        key: &[Word; 8],
        offsets: &[u64],
        flags_all: Word,
        flags_start: Word,
        flags_end: Word,
        out: &mut [u8],
    ) {
        let blocks = CHUNK_BYTES / BLOCK_BYTES;
        unsafe {
            // Safe because the layout of arrays is guaranteed, and because the
            // `blocks` count determined statically from the argument type.
            let chunk_ptrs: &[*const u8] =
                core::slice::from_raw_parts(chunks.as_ptr() as *const *const u8, chunks.len());
            // Safe because detect() checked for platform support.
            (self.hash_many_fn)(
                chunk_ptrs,
                blocks,
                key,
                offsets,
                flags_all,
                flags_start,
                flags_end,
                out,
            );
        }
    }
}

unsafe fn hash_one(
    input: *const u8,
    mut blocks: usize,
    key: &[Word; 8],
    offset: u64,
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    out: &mut [u8; OUT_BYTES],
    compression_fn: CompressionFn,
) {
    let mut state = iv(key);
    let mut flags = flags_all | flags_start;
    while blocks > 0 {
        if blocks == 1 {
            flags |= flags_end;
        }
        compression_fn(
            &mut state,
            &*(input as *const [u8; BLOCK_BYTES]),
            BLOCK_BYTES as Word,
            offset,
            flags,
        );
        blocks -= 1;
        flags = flags_all;
    }
    *out = bytes_from_state_words(&state);
}

unsafe fn hash_many_serial(
    mut inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    mut offsets: &[u64],
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
    compression_fn: CompressionFn,
) {
    while !inputs.is_empty() && !offsets.is_empty() && out.len() >= OUT_BYTES {
        hash_one(
            inputs[0],
            blocks,
            key,
            offsets[0],
            flags_all,
            flags_start,
            flags_end,
            array_mut_ref!(out, 0, OUT_BYTES),
            compression_fn,
        );
        inputs = &inputs[1..];
        offsets = &offsets[1..];
        out = &mut out[OUT_BYTES..];
    }
}

unsafe fn hash_many_portable(
    inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    offsets: &[u64],
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    out: &mut [u8],
) {
    hash_many_serial(
        inputs,
        blocks,
        key,
        offsets,
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
    mut offsets: &[u64],
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
) {
    while inputs.len() >= sse41::DEGREE
        && offsets.len() >= sse41::DEGREE
        && out.len() >= sse41::DEGREE * OUT_BYTES
    {
        let inputs_array = array_ref!(inputs, 0, sse41::DEGREE);
        let offsets_array = array_ref!(offsets, 0, sse41::DEGREE);
        let out_array = array_mut_ref!(out, 0, sse41::DEGREE * OUT_BYTES);
        sse41::compress4_loop(
            inputs_array,
            blocks,
            key,
            offsets_array,
            flags_all,
            flags_start,
            flags_end,
            out_array,
        );
        inputs = &inputs[sse41::DEGREE..];
        offsets = &offsets[sse41::DEGREE..];
        out = &mut out[sse41::DEGREE * OUT_BYTES..];
    }

    while !inputs.is_empty() && !offsets.is_empty() && out.len() >= OUT_BYTES {
        hash_one(
            inputs[0],
            blocks,
            key,
            offsets[0],
            flags_all,
            flags_start,
            flags_end,
            array_mut_ref!(out, 0, OUT_BYTES),
            sse41::compress,
        );
        inputs = &inputs[1..];
        offsets = &offsets[1..];
        out = &mut out[OUT_BYTES..];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hash_many_avx2(
    mut inputs: &[*const u8],
    blocks: usize,
    key: &[Word; 8],
    mut offsets: &[u64],
    flags_all: Word,
    flags_start: Word,
    flags_end: Word,
    mut out: &mut [u8],
) {
    while inputs.len() >= avx2::DEGREE
        && offsets.len() >= avx2::DEGREE
        && out.len() >= avx2::DEGREE * OUT_BYTES
    {
        let inputs_array = array_ref!(inputs, 0, avx2::DEGREE);
        let offsets_array = array_ref!(offsets, 0, avx2::DEGREE);
        let out_array = array_mut_ref!(out, 0, avx2::DEGREE * OUT_BYTES);
        avx2::compress8_loop(
            inputs_array,
            blocks,
            key,
            offsets_array,
            flags_all,
            flags_start,
            flags_end,
            out_array,
        );
        inputs = &inputs[avx2::DEGREE..];
        offsets = &offsets[avx2::DEGREE..];
        out = &mut out[avx2::DEGREE * OUT_BYTES..];
    }

    hash_many_sse41(
        inputs,
        blocks,
        key,
        offsets,
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

        let mut input_buf = [0; NUM_INPUTS * BLOCK_BYTES];
        crate::test::paint_test_input(&mut input_buf);
        let mut blocks_array = ArrayVec::<[*const u8; NUM_INPUTS]>::new();
        let mut offsets_array = ArrayVec::<[u64; NUM_INPUTS]>::new();
        for i in 0..NUM_INPUTS {
            blocks_array.push(input_buf[i * BLOCK_BYTES..].as_ptr());
            offsets_array.push(i as u64);
        }

        let mut out_portable = [0; NUM_INPUTS * OUT_BYTES];
        unsafe {
            hash_many_portable(
                &blocks_array,
                1,
                &[42; 8],
                &offsets_array,
                1,
                2,
                4,
                &mut out_portable,
            );
        }

        let mut out_sse41 = [0; NUM_INPUTS * OUT_BYTES];
        unsafe {
            hash_many_sse41(
                &blocks_array,
                1,
                &[42; 8],
                &offsets_array,
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

        let mut input_buf = [0; NUM_INPUTS * BLOCK_BYTES];
        crate::test::paint_test_input(&mut input_buf);
        let mut blocks_array = ArrayVec::<[*const u8; NUM_INPUTS]>::new();
        let mut offsets_array = ArrayVec::<[u64; NUM_INPUTS]>::new();
        for i in 0..NUM_INPUTS {
            blocks_array.push(input_buf[i * BLOCK_BYTES..].as_ptr());
            offsets_array.push(i as u64);
        }

        let mut out_portable = [0; NUM_INPUTS * OUT_BYTES];
        unsafe {
            hash_many_portable(
                &blocks_array,
                1,
                &[42; 8],
                &offsets_array,
                1,
                2,
                4,
                &mut out_portable,
            );
        }

        let mut out_avx2 = [0; NUM_INPUTS * OUT_BYTES];
        unsafe {
            hash_many_avx2(
                &blocks_array,
                1,
                &[42; 8],
                &offsets_array,
                1,
                2,
                4,
                &mut out_avx2,
            );
        }

        assert_eq!(&out_portable[..], &out_avx2[..]);
    }
}
