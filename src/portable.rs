use crate::{Word, BLOCK_LEN, IV, MSG_SCHEDULE, WORD_BITS};
use arrayref::array_refs;

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
    block_len: Word,
    offset: u64,
    flags: Word,
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
        IV[4] ^ offset as Word,
        IV[5] ^ (offset >> WORD_BITS) as Word,
        IV[6] ^ block_len,
        IV[7] ^ flags,
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
