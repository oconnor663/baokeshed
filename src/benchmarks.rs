//! Hidden public interfaces that are only intended for benchmarks. These are
//! not stable. If you find yourself needing these in production, please file a
//! GitHub issue.

pub fn compress_portable(
    state: &mut [crate::Word; 8],
    block: &[u8; crate::BLOCK_LEN],
    block_len: u8,
    offset: u64,
    flags: u8,
) {
    crate::portable::compress(state, block, block_len, offset, flags)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn compress_sse41(
    state: &mut [crate::Word; 8],
    block: &[u8; crate::BLOCK_LEN],
    block_len: u8,
    offset: u64,
    flags: u8,
) {
    crate::sse41::compress(state, block, block_len, offset, flags)
}
