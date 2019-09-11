use arrayref::{array_mut_ref, array_ref, array_refs, mut_array_refs};
use arrayvec::ArrayVec;
use core::cmp;
use core::mem::size_of;

type Word = u32;

const WORD_BYTES: usize = size_of::<Word>();
const WORD_BITS: usize = 8 * WORD_BYTES;
pub const OUT_BYTES: usize = 8 * WORD_BYTES;
pub const KEY_BYTES: usize = 8 * WORD_BYTES;
pub const BLOCK_BYTES: usize = 16 * WORD_BYTES;
pub const CHUNK_BYTES: usize = 4096;

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
    count: u64,
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
        IV[5] ^ (count >> WORD_BITS) as Word,
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

fn words_from_msg_bytes(bytes: &[u8; BLOCK_BYTES]) -> [Word; 16] {
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

fn words_from_key_bytes(bytes: &[u8; KEY_BYTES]) -> [Word; 8] {
    // Parse the message bytes as little endian words.
    const W: usize = size_of::<Word>();
    let refs = array_refs!(bytes, W, W, W, W, W, W, W, W);
    [
        Word::from_le_bytes(*refs.0),
        Word::from_le_bytes(*refs.1),
        Word::from_le_bytes(*refs.2),
        Word::from_le_bytes(*refs.3),
        Word::from_le_bytes(*refs.4),
        Word::from_le_bytes(*refs.5),
        Word::from_le_bytes(*refs.6),
        Word::from_le_bytes(*refs.7),
    ]
}

fn bytes_from_state_words(words: &[Word; 8]) -> [u8; OUT_BYTES] {
    let mut bytes = [0; OUT_BYTES];
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
    buf: [u8; BLOCK_BYTES],
    buf_len: u8,
    // Note count begins at the chunk's starting offset, not zero.
    count: u64,
}

impl ChunkState {
    fn new(key: &[Word; 8], count: u64) -> Self {
        debug_assert_eq!(count % CHUNK_BYTES as u64, 0);
        Self {
            half_state: iv(key),
            buf: [0; BLOCK_BYTES],
            buf_len: 0,
            count,
        }
    }

    // The length of the current chunk, including buffered bytes.
    fn len(&self) -> usize {
        // The count never fully rolls over to the start of the next chunk, so
        // we don't have to worry about this remainder wrapping to 0.
        let count_this_chunk = self.count % CHUNK_BYTES as u64;
        let len = count_this_chunk as usize + self.buf_len as usize;
        debug_assert!(len <= CHUNK_BYTES);
        len
    }

    // The total count of all input bytes so far, including buffered bytes and
    // previous chunks.
    fn total_count(&self) -> u64 {
        // Note we keep incrementing the count across chunks, so the count of
        // the current chunk reflects the input bytes hashed by all chunks so
        // far.
        self.count + self.buf_len as u64
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let want = BLOCK_BYTES - self.buf_len as usize;
        let take = cmp::min(want, input.len());
        self.buf[self.buf_len as usize..self.buf_len as usize + take]
            .copy_from_slice(&input[..take]);
        self.buf_len += take as u8;
        *input = &input[take..];
    }

    fn append(&mut self, mut input: &[u8]) {
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                debug_assert_eq!(self.buf_len as usize, BLOCK_BYTES);
                let block_words = words_from_msg_bytes(&self.buf);
                let flags = 0; // TODO
                compress(
                    &mut self.half_state,
                    &block_words,
                    self.count,
                    BLOCK_BYTES as Word,
                    flags,
                );
                self.count += BLOCK_BYTES as u64;
                self.buf_len = 0;
                self.buf = [0; BLOCK_BYTES];
            }
        }

        while input.len() > BLOCK_BYTES {
            debug_assert_eq!(self.buf_len, 0);
            let block = array_ref!(input, 0, BLOCK_BYTES);
            let block_words = words_from_msg_bytes(block);
            let flags = 0; // TODO
            compress(
                &mut self.half_state,
                &block_words,
                self.count,
                BLOCK_BYTES as Word,
                flags,
            );
            self.count += BLOCK_BYTES as u64;
            input = &input[BLOCK_BYTES..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
        debug_assert!(self.len() <= CHUNK_BYTES);
    }

    fn finalize(&self) -> [Word; 8] {
        let mut output = self.half_state;
        let block_words = words_from_msg_bytes(&self.buf);
        let flags = 0; // TODO
        compress(
            &mut output,
            &block_words,
            self.count,
            self.buf_len as Word,
            flags,
        );
        output
    }
}

fn hash_parent(left_hash: &[Word; 8], right_hash: &[Word; 8], key: &[Word; 8]) -> [Word; 8] {
    let parent_words = [
        left_hash[0],
        left_hash[1],
        left_hash[2],
        left_hash[3],
        left_hash[4],
        left_hash[5],
        left_hash[6],
        left_hash[7],
        right_hash[0],
        right_hash[1],
        right_hash[2],
        right_hash[3],
        right_hash[4],
        right_hash[5],
        right_hash[6],
        right_hash[7],
    ];
    let mut state = iv(key);
    let flags = 0; // TODO
    compress(&mut state, &parent_words, 0, BLOCK_BYTES as Word, flags);
    state
}

pub struct Hasher {
    key: [Word; 8],
    chunk: ChunkState,
    // This array is bigger than it needs to be (2 KiB instead of 1.7 KiB),
    // because arrayvec doesn't currently support a length of 52 * 8 = 416. We
    // could shrink this with a custom ArrayVec using unsafe code, but stable
    // cont generics will make everything better someday.
    subtree_hash_words: ArrayVec<[Word; 512]>,
}

impl Hasher {
    pub fn new(key_bytes: &[u8; KEY_BYTES]) -> Self {
        let key = words_from_key_bytes(key_bytes);
        Self {
            key,
            chunk: ChunkState::new(&key, 0),
            subtree_hash_words: ArrayVec::new(),
        }
    }

    fn num_subtrees(&self) -> usize {
        debug_assert_eq!(self.subtree_hash_words.len() % 8, 0);
        self.subtree_hash_words.len() / 8
    }

    // We keep subtree hashes packed in the subtree_hash_words array without
    // storing subtree sizes anywhere, and we use this cute trick to figure out
    // when we should merge them. Because every subtree (prior to the
    // finalization step) is a power of two times the chunk size, adding a new
    // subtree to the right/small end is a lot like adding a 1 to a binary
    // number, and merging subtrees is like propagating the carry bit. Each
    // carry represents a place where two subtrees need to be merged, and the
    // final number of 1 bits is the same as the final number of subtrees.
    fn needs_merge(&self) -> bool {
        debug_assert_eq!(self.chunk.total_count() % CHUNK_BYTES as u64, 0);
        let total_chunks = self.chunk.total_count() / CHUNK_BYTES as u64;
        self.num_subtrees() > total_chunks.count_ones() as usize
    }

    // Take two subtree hashes off the end of the stack, hash them into a
    // parent node, and put that hash back on the stack.
    fn merge_parent(&mut self) {
        debug_assert!(self.num_subtrees() >= 2);
        let left_start = self.subtree_hash_words.len() - 16;
        let right_start = self.subtree_hash_words.len() - 8;
        let left_hash = array_ref!(self.subtree_hash_words, left_start, 8);
        let right_hash = array_ref!(self.subtree_hash_words, right_start, 8);
        let hash = hash_parent(left_hash, right_hash, &self.key);
        *array_mut_ref!(self.subtree_hash_words, left_start, 8) = hash;
        let new_len = self.subtree_hash_words.len() - 8;
        self.subtree_hash_words.truncate(new_len)
    }

    pub fn append(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_BYTES {
                let chunk_hash = self.chunk.finalize();
                self.subtree_hash_words.extend(chunk_hash.iter().copied());
                self.chunk = ChunkState::new(&self.key, self.chunk.total_count());
                while self.needs_merge() {
                    self.merge_parent();
                }
            }
            let want = CHUNK_BYTES - self.chunk.len();
            let take = cmp::min(want, input.len());
            self.chunk.append(&input[..take]);
            input = &input[take..];
        }
    }

    pub fn finalize(&self) -> [Word; 8] {
        // Finalize the current chunk.
        // TODO: root finalization flags
        let mut hash = self.chunk.finalize();
        // Merge that rightmost chunk hash with every hash in the subtree
        // stack, from right (smallest) to left (largest) to produce the final
        // root hash.
        for subtree in self.subtree_hash_words.chunks_exact(8).rev() {
            hash = hash_parent(array_ref!(subtree, 0, 8), &hash, &self.key);
        }
        hash
    }
}
