use crate::{block_flags, offset_high, offset_low, Word, BLOCK_LEN, IV, MSG_SCHEDULE, OUT_LEN};
use arrayref::{array_mut_ref, array_ref, array_refs};

#[inline(always)]
fn words_from_block(bytes: &[u8; BLOCK_LEN]) -> [Word; 16] {
    let refs = array_refs!(bytes, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
    [
        Word::from_le_bytes(*refs.0),
        Word::from_le_bytes(*refs.1),
        Word::from_le_bytes(*refs.2),
        Word::from_le_bytes(*refs.3),
        Word::from_le_bytes(*refs.4),
        Word::from_le_bytes(*refs.5),
        Word::from_le_bytes(*refs.6),
        Word::from_le_bytes(*refs.7),
        Word::from_le_bytes(*refs.8),
        Word::from_le_bytes(*refs.9),
        Word::from_le_bytes(*refs.10),
        Word::from_le_bytes(*refs.11),
        Word::from_le_bytes(*refs.12),
        Word::from_le_bytes(*refs.13),
        Word::from_le_bytes(*refs.14),
        Word::from_le_bytes(*refs.15),
    ]
}

#[inline(always)]
fn g(state: &mut [Word; 16], a: usize, b: usize, c: usize, d: usize, x: Word, y: Word) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [Word; 16], msg: &[Word; 16], round: usize) {
    // Select the message schedule based on the round.
    let schedule = MSG_SCHEDULE[round];

    // Mix the columns.
    g(state, 0, 4, 8, 12, msg[schedule[0]], msg[schedule[1]]);
    g(state, 1, 5, 9, 13, msg[schedule[2]], msg[schedule[3]]);
    g(state, 2, 6, 10, 14, msg[schedule[4]], msg[schedule[5]]);
    g(state, 3, 7, 11, 15, msg[schedule[6]], msg[schedule[7]]);

    // Mix the rows.
    g(state, 0, 5, 10, 15, msg[schedule[8]], msg[schedule[9]]);
    g(state, 1, 6, 11, 12, msg[schedule[10]], msg[schedule[11]]);
    g(state, 2, 7, 8, 13, msg[schedule[12]], msg[schedule[13]]);
    g(state, 3, 4, 9, 14, msg[schedule[14]], msg[schedule[15]]);
}

pub fn compress(
    state: &mut [Word; 8],
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    offset: u64,
    internal_flags: u8,
    context: Word,
) {
    let block_words = words_from_block(block);
    let mut full_state = [
        state[0],
        state[1],
        state[2],
        state[3],
        state[4],
        state[5],
        state[6],
        state[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ offset_low(offset),
        IV[5] ^ offset_high(offset),
        IV[6] ^ block_flags(block_len, internal_flags),
        IV[7] ^ context,
    ];

    round(&mut full_state, &block_words, 0);
    round(&mut full_state, &block_words, 1);
    round(&mut full_state, &block_words, 2);
    round(&mut full_state, &block_words, 3);
    round(&mut full_state, &block_words, 4);
    round(&mut full_state, &block_words, 5);
    round(&mut full_state, &block_words, 6);

    state[0] = full_state[0] ^ full_state[8];
    state[1] = full_state[1] ^ full_state[9];
    state[2] = full_state[2] ^ full_state[10];
    state[3] = full_state[3] ^ full_state[11];
    state[4] = full_state[4] ^ full_state[12];
    state[5] = full_state[5] ^ full_state[13];
    state[6] = full_state[6] ^ full_state[14];
    state[7] = full_state[7] ^ full_state[15];
}

pub fn hash1<A: arrayvec::Array<Item = u8>>(
    input: &A,
    key: &[Word; 8],
    offset: u64,
    internal_flags_start: u8,
    internal_flags_end: u8,
    context: Word,
    out: &mut [u8; OUT_LEN],
) {
    debug_assert_eq!(A::CAPACITY % BLOCK_LEN, 0, "uneven blocks");
    let mut state = crate::iv(key);
    let mut internal_flags = internal_flags_start;
    let mut slice = input.as_slice();
    while slice.len() >= BLOCK_LEN {
        if slice.len() == BLOCK_LEN {
            internal_flags |= internal_flags_end;
        }
        compress(
            &mut state,
            array_ref!(slice, 0, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            internal_flags,
            context,
        );
        internal_flags = 0;
        slice = &slice[BLOCK_LEN..];
    }
    *out = crate::bytes_from_state_words(&state);
}

pub fn hash_many<A: arrayvec::Array<Item = u8>>(
    inputs: &[&A],
    key: &[Word; 8],
    mut offset: u64,
    offset_delta: u64,
    internal_flags_start: u8,
    internal_flags_end: u8,
    context: Word,
    out: &mut [u8],
) {
    debug_assert!(out.len() >= inputs.len() * OUT_LEN, "out too short");
    for (&input, output) in inputs.iter().zip(out.chunks_exact_mut(OUT_LEN)) {
        hash1(
            input,
            key,
            offset,
            internal_flags_start,
            internal_flags_end,
            context,
            array_mut_ref!(output, 0, OUT_LEN),
        );
        offset += offset_delta;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_hash1_1() {
        let block = [1; BLOCK_LEN];
        let key = [2; 8];
        let offset = 3 * crate::CHUNK_LEN as u64;
        let flags_all = 0; // currently unused
        let flags_start = 4;
        let flags_end = 5;
        let context = 6;

        let mut expected_state = crate::iv(&key);
        compress(
            &mut expected_state,
            &block,
            BLOCK_LEN as u8,
            offset,
            flags_all | flags_start | flags_end,
            context,
        );
        let expected_out = crate::bytes_from_state_words(&expected_state);

        let mut test_out = [0; OUT_LEN];
        hash1(
            &block,
            &key,
            offset,
            flags_start,
            flags_end,
            context,
            &mut test_out,
        );

        assert_eq!(expected_out, test_out);
    }

    #[test]
    fn test_hash1_3() {
        let mut blocks = [0; BLOCK_LEN * 3];
        crate::test::paint_test_input(&mut blocks);
        let key = [2; 8];
        let offset = 3 * crate::CHUNK_LEN as u64;
        let flags_all = 0; // currently unused
        let flags_start = 4;
        let flags_end = 5;
        let context = 6;

        let mut expected_state = crate::iv(&key);
        compress(
            &mut expected_state,
            array_ref!(blocks, 0, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            flags_all | flags_start,
            context,
        );
        compress(
            &mut expected_state,
            array_ref!(blocks, BLOCK_LEN, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            flags_all,
            context,
        );
        compress(
            &mut expected_state,
            array_ref!(blocks, 2 * BLOCK_LEN, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            flags_all | flags_end,
            context,
        );
        let expected_out = crate::bytes_from_state_words(&expected_state);

        let mut test_out = [0; OUT_LEN];
        hash1(
            &blocks,
            &key,
            offset,
            flags_start,
            flags_end,
            context,
            &mut test_out,
        );

        assert_eq!(expected_out, test_out);
    }

    #[test]
    fn test_hash_many() {
        let mut input_buf = [0; BLOCK_LEN * 9];
        crate::test::paint_test_input(&mut input_buf);
        let inputs = [
            array_ref!(input_buf, 0 * BLOCK_LEN, 3 * BLOCK_LEN),
            array_ref!(input_buf, 3 * BLOCK_LEN, 3 * BLOCK_LEN),
            array_ref!(input_buf, 6 * BLOCK_LEN, 3 * BLOCK_LEN),
        ];
        let key = [2; 8];
        let offset = 3 * crate::CHUNK_LEN as u64;
        let delta = 4;
        let flags_start = 5;
        let flags_end = 6;
        let context = 7;

        let mut expected_out = [0; 3 * OUT_LEN];
        hash1(
            inputs[0],
            &key,
            offset + 0 * delta,
            flags_start,
            flags_end,
            context,
            array_mut_ref!(&mut expected_out, 0 * OUT_LEN, OUT_LEN),
        );
        hash1(
            inputs[1],
            &key,
            offset + 1 * delta,
            flags_start,
            flags_end,
            context,
            array_mut_ref!(&mut expected_out, 1 * OUT_LEN, OUT_LEN),
        );
        hash1(
            inputs[2],
            &key,
            offset + 2 * delta,
            flags_start,
            flags_end,
            context,
            array_mut_ref!(&mut expected_out, 2 * OUT_LEN, OUT_LEN),
        );

        let mut test_out = [0; OUT_LEN * 3];
        hash_many(
            &inputs,
            &key,
            offset,
            delta,
            flags_start,
            flags_end,
            context,
            &mut test_out,
        );

        assert_eq!(&expected_out[..], &test_out[..]);
    }
}
