use crate::{Word, BLOCK_LEN};

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

mod ffi {
    extern "C" {
        pub fn compress(
            state: *mut u32,
            block: *const u8,
            block_len: u8,
            offset: u64,
            internal_flags: u8,
            context: u32,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{CHUNK_LEN, WORD_BITS};

    #[test]
    fn compare_portable() {
        let initial_state = [1, 2, 3, 4, 5, 6, 7, 8];
        let block_len: u8 = 27;
        let mut block = [0; BLOCK_LEN];
        crate::test::paint_test_input(&mut block[..block_len as usize]);
        // Use an offset with set bits in both 32-bit words.
        let offset = ((5 * CHUNK_LEN as u64) << WORD_BITS) + 6 * CHUNK_LEN as u64;
        let flags = crate::Flags::CHUNK_END | crate::Flags::ROOT;
        let context = 23;

        let mut portable_state = initial_state;
        crate::portable::compress(
            &mut portable_state,
            &block,
            block_len,
            offset as u64,
            flags.bits(),
            context,
        );

        let mut ffi_state = initial_state;
        super::compress(
            &mut ffi_state,
            &block,
            block_len,
            offset as u64,
            flags.bits(),
            context,
        );

        assert_eq!(portable_state, ffi_state);
    }
}
