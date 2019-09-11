use arrayref::{array_ref, array_refs, mut_array_refs};
use core::cmp;
use core::mem::size_of;

type Word = u32;
type Count = u64;

const WORDBYTES: usize = size_of::<Word>();
const WORDBITS: usize = 8 * WORDBYTES;
pub const OUTBYTES: usize = 8 * WORDBYTES;
pub const KEYBYTES: usize = 8 * WORDBYTES;
pub const BLOCKBYTES: usize = 16 * WORDBYTES;
pub const CHUNKBYTES: usize = 4096;

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

fn compress(
    half_state: &mut [Word; 8],
    block_words: &[Word; 16],
    count: Count,
    block_len: Word,
    flags: Word,
) {
    let mut full_state = [
        half_state[0],
        half_state[1],
        half_state[2],
        half_state[3],
        half_state[4],
        half_state[5],
        half_state[6],
        half_state[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ count as Word,
        IV[5] ^ (count >> WORDBITS) as Word,
        IV[6] ^ block_len,
        IV[7] ^ flags,
    ];

    round(&mut full_state, block_words, 0);
    round(&mut full_state, block_words, 1);
    round(&mut full_state, block_words, 2);
    round(&mut full_state, block_words, 3);
    round(&mut full_state, block_words, 4);
    round(&mut full_state, block_words, 5);
    round(&mut full_state, block_words, 6);

    half_state[0] ^= full_state[0] ^ full_state[8];
    half_state[1] ^= full_state[1] ^ full_state[9];
    half_state[2] ^= full_state[2] ^ full_state[10];
    half_state[3] ^= full_state[3] ^ full_state[11];
    half_state[4] ^= full_state[4] ^ full_state[12];
    half_state[5] ^= full_state[5] ^ full_state[13];
    half_state[6] ^= full_state[6] ^ full_state[14];
    half_state[7] ^= full_state[7] ^ full_state[15];
}

fn words_from_bytes(bytes: &[u8; BLOCKBYTES]) -> [Word; 16] {
    // Parse the message bytes as little endian words.
    const W: usize = size_of::<Word>();
    let refs = array_refs!(bytes, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W);
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

fn bytes_from_words(words: &[Word; 8]) -> [u8; OUTBYTES] {
    let mut bytes = [0; OUTBYTES];
    {
        const W: usize = size_of::<Word>();
        let refs = mut_array_refs!(&mut bytes, W, W, W, W, W, W, W, W);
        *refs.0 = words[0].to_le_bytes();
        *refs.1 = words[1].to_le_bytes();
        *refs.2 = words[2].to_le_bytes();
        *refs.3 = words[3].to_le_bytes();
        *refs.4 = words[4].to_le_bytes();
        *refs.5 = words[5].to_le_bytes();
        *refs.6 = words[6].to_le_bytes();
        *refs.7 = words[7].to_le_bytes();
    }
    bytes
}

fn iv(key: &[Word; 8]) -> [Word; 8] {
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

struct ChunkState {
    half_state: [Word; 8],
    buf: [u8; BLOCKBYTES],
    buf_len: u8,
    count: Count,
}

impl ChunkState {
    fn new(key: &[Word; 8], count: Count) -> Self {
        Self {
            half_state: iv(key),
            buf: [0; BLOCKBYTES],
            buf_len: 0,
            count,
        }
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let take = cmp::min(BLOCKBYTES - self.buf_len as usize, input.len());
        self.buf[self.buf_len as usize..self.buf_len as usize + take]
            .copy_from_slice(&input[..take]);
        self.buf_len += take as u8;
        *input = &input[take..];
    }

    fn update(&mut self, mut input: &[u8]) {
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                debug_assert_eq!(self.buf_len as usize, BLOCKBYTES);
                let block_words = words_from_bytes(&self.buf);
                let flags = 0; // TODO
                compress(
                    &mut self.half_state,
                    &block_words,
                    self.count,
                    BLOCKBYTES as Word,
                    flags,
                );
                self.count += BLOCKBYTES as u64;
                self.buf_len = 0;
                self.buf = [0; BLOCKBYTES];
            }
        }

        while input.len() > BLOCKBYTES {
            debug_assert_eq!(self.buf_len, 0);
            let block = array_ref!(input, 0, BLOCKBYTES);
            let block_words = words_from_bytes(block);
            let flags = 0; // TODO
            compress(
                &mut self.half_state,
                &block_words,
                self.count,
                BLOCKBYTES as Word,
                flags,
            );
            self.count += BLOCKBYTES as u64;
            input = &input[BLOCKBYTES..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
    }

    fn finalize(&self) -> [Word; 8] {
        let mut state_copy = self.half_state;
        let block_words = words_from_bytes(&self.buf);
        let flags = 0; // TODO
        compress(
            &mut state_copy,
            &block_words,
            self.count,
            self.buf_len as Word,
            flags,
        );
        state_copy
    }
}
