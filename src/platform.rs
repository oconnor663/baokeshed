use crate::{portable, Word, BLOCK_LEN};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::{avx2, sse41};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const MAX_SIMD_DEGREE: usize = 8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub const MAX_SIMD_DEGREE: usize = 1;

// There are some places where we want a static size that's equal to the
// MAX_SIMD_DEGREE, but also at least 2. Constant contexts aren't currently
// allowed to use cmp::max, so we have to hardcode this additional constant
// value. Get rid of this once cmp::max is a const fn.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub const MAX_SIMD_DEGREE_OR_2: usize = 8;
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub const MAX_SIMD_DEGREE_OR_2: usize = 2;

#[derive(Clone, Copy)]
pub enum Platform {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    SSE41,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    AVX2,
}

impl Platform {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return Platform::AVX2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return Platform::SSE41;
            }
        }
        Platform::Portable
    }

    pub fn simd_degree(&self) -> usize {
        match self {
            Platform::Portable => 1,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => 4,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => 8,
        }
    }

    pub fn compress(
        &self,
        cv: &mut [Word; 8],
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        offset: u64,
        flags: u8,
    ) {
        match self {
            Platform::Portable => portable::compress(cv, block, block_len, offset, flags),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 | Platform::AVX2 => unsafe {
                sse41::compress(cv, block, block_len, offset, flags)
            },
        }
    }

    pub fn compress_xof(
        &self,
        cv: &[Word; 8],
        block: &[u8; BLOCK_LEN],
        block_len: u8,
        offset: u64,
        flags: u8,
    ) -> [u8; 64] {
        match self {
            Platform::Portable => portable::compress_xof(cv, block, block_len, offset, flags),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 | Platform::AVX2 => unsafe {
                sse41::compress_xof(cv, block, block_len, offset, flags)
            },
        }
    }

    pub fn hash_many<A: arrayvec::Array<Item = u8>>(
        &self,
        inputs: &[&A],
        key: &[Word; 8],
        offset: u64,
        offset_delta: u64,
        flags: u8,
        flags_start: u8,
        flags_end: u8,
        out: &mut [u8],
    ) {
        match self {
            Platform::Portable => portable::hash_many(
                inputs,
                key,
                offset,
                offset_delta,
                flags,
                flags_start,
                flags_end,
                out,
            ),
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::SSE41 => unsafe {
                sse41::hash_many(
                    inputs,
                    key,
                    offset,
                    offset_delta,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
            // Safe because detect() checked for platform support.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Platform::AVX2 => unsafe {
                avx2::hash_many(
                    inputs,
                    key,
                    offset,
                    offset_delta,
                    flags,
                    flags_start,
                    flags_end,
                    out,
                )
            },
        }
    }
}
