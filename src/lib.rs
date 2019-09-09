use arrayref::array_refs;
use core::mem::size_of;

type Word = u32;
type Count = u64;

const WORD_BYTES: usize = size_of::<Word>();
const WORD_BITS: usize = 8 * WORD_BYTES;
pub const OUTBYTES: usize = 8 * WORD_BYTES;
pub const KEYBYTES: usize = 8 * WORD_BYTES;
pub const BLOCKBYTES: usize = 16 * WORD_BYTES;

const IV: [Word; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
];

#[inline(always)]
fn g(v: &mut [Word; 16], a: usize, b: usize, c: usize, d: usize, x: Word, y: Word) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

#[inline(always)]
fn round(round: usize, msg: &[Word; 16], state: &mut [Word; 16]) {
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

#[inline(always)]
fn compress(
    state: &mut [Word; 16],
    msg: &[Word; 16],
    count: Count,
    parent_flag: Word,
    kdf_flag: Word,
) {
    // The first half of the state is the chaining value. Overwrite the second
    // half of the state with new initial values.
    state[0] = IV[0];
    state[1] = IV[1];
    state[2] = IV[2];
    state[3] = IV[3];
    state[4] = IV[4] ^ count as Word;
    state[5] = IV[5] ^ (count >> WORD_BITS) as Word;
    state[6] = IV[6] ^ parent_flag;
    state[7] = IV[7] ^ kdf_flag;

    round(0, msg, state);
    round(1, msg, state);
    round(2, msg, state);
    round(3, msg, state);
    round(4, msg, state);
    round(5, msg, state);
    round(6, msg, state);
}

#[inline(always)]
fn initial_cv(key: &[Word; 8]) -> [Word; 8] {
    [
        IV[0] ^ key[0],
        IV[1] ^ key[1],
        IV[2] ^ key[2],
        IV[3] ^ key[3],
        IV[4] ^ key[4],
        IV[5] ^ key[5],
        IV[6] ^ key[6],
        IV[7] ^ key[7],
    ]
}

#[inline(always)]
fn initial_state(key: &[Word; 8]) -> [Word; 16] {
    let cv = initial_cv(key);
    [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        // The rest gets set by the compression function.
        0, 0, 0, 0, 0, 0, 0, 0,
    ]
}

#[inline(always)]
fn hash_parent(
    left_cv: &[Word; 8],
    right_cv: &[Word; 8],
    key: &[Word; 8],
    kdf_flag: Word,
) -> [Word; 16] {
    let msg = [
        left_cv[0],
        left_cv[1],
        left_cv[2],
        left_cv[3],
        left_cv[4],
        left_cv[5],
        left_cv[6],
        left_cv[7],
        right_cv[0],
        right_cv[1],
        right_cv[2],
        right_cv[3],
        right_cv[4],
        right_cv[5],
        right_cv[6],
        right_cv[7],
    ];

    let mut state = initial_state(key);
    compress(&mut state, &msg, 0, !0, kdf_flag);
    state
}
