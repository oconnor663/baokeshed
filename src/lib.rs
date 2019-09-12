use arrayref::{array_mut_ref, array_ref, array_refs, mut_array_refs};
use arrayvec::ArrayVec;
use core::cmp;
use core::fmt;
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

// The CHUNK_END and PARENT flags might not be strictly necessary. CHUNK_END
// might not be necessary because the keyed root node makes it impossible to
// extend a chunk hash without the key. PARENT might not be necessary because
// CHUNK_START already effectively domain-separates chunks from parent nodes.
// But it doesn't cost anything to be conservative here.
bitflags::bitflags! {
    struct Flags: Word {
        const CHUNK_START = 1;
        const CHUNK_END = 2;
        const PARENT = 4;
        const ROOT = 8;
    }
}

enum IsRoot {
    NotRoot,
    Root,
}

impl IsRoot {
    fn flag(&self) -> Flags {
        match self {
            IsRoot::NotRoot => Flags::empty(),
            IsRoot::Root => Flags::ROOT,
        }
    }
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

fn compress(
    half_state: &mut [Word; 8],
    block_words: &[Word; 16],
    count: u64,
    block_len: Word,
    flags: Flags,
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
        IV[7] ^ flags.bits(),
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

// =======================================================================
// ================== Recursive tree hash implementation =================
// =======================================================================

fn hash_chunk(mut chunk: &[u8], key: &[Word; 8], offset: u64, is_root: IsRoot) -> [Word; 8] {
    debug_assert!(chunk.len() <= CHUNK_BYTES);
    debug_assert_eq!(offset % CHUNK_BYTES as u64, 0);
    let mut state = iv(key);
    let mut count = offset;
    let mut maybe_start_flag = Flags::CHUNK_START;
    while chunk.len() > BLOCK_BYTES {
        let block = words_from_msg_bytes(array_ref!(chunk, 0, BLOCK_BYTES));
        compress(
            &mut state,
            &block,
            count,
            BLOCK_BYTES as Word,
            maybe_start_flag,
        );
        count += BLOCK_BYTES as u64;
        chunk = &chunk[BLOCK_BYTES..];
        maybe_start_flag = Flags::empty();
    }
    let mut last_block = [0; BLOCK_BYTES];
    last_block[..chunk.len()].copy_from_slice(chunk);
    compress(
        &mut state,
        &words_from_msg_bytes(&last_block),
        count,
        chunk.len() as Word,
        maybe_start_flag | Flags::CHUNK_END | is_root.flag(),
    );
    state
}

fn hash_parent(
    left_hash: &[Word; 8],
    right_hash: &[Word; 8],
    key: &[Word; 8],
    is_root: IsRoot,
) -> [Word; 8] {
    let mut parent_words = [0; 16];
    let refs = mut_array_refs!(&mut parent_words, 8, 8);
    *refs.0 = *left_hash;
    *refs.1 = *right_hash;
    let mut state = iv(key);
    compress(
        &mut state,
        &parent_words,
        0,
        BLOCK_BYTES as Word,
        Flags::PARENT | is_root.flag(),
    );
    state
}

// Find the largest power of two that's less than or equal to `n`. We use this
// for computing subtree sizes below.
fn largest_power_of_two_leq(n: usize) -> usize {
    ((n / 2) + 1).next_power_of_two()
}

// Given some input larger than one chunk, find the largest full tree of chunks
// that can go on the left.
fn left_len(content_len: usize) -> usize {
    debug_assert!(content_len > CHUNK_BYTES);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / CHUNK_BYTES;
    largest_power_of_two_leq(full_chunks) * CHUNK_BYTES
}

fn hash_recurse(input: &[u8], key: &[Word; 8], count: u64, is_root: IsRoot) -> [Word; 8] {
    if input.len() <= CHUNK_BYTES {
        return hash_chunk(input, key, count, is_root);
    }
    let (left, right) = input.split_at(left_len(input.len()));
    let left_hash = hash_recurse(left, key, count, IsRoot::NotRoot);
    let right_hash = hash_recurse(right, key, count + left.len() as u64, IsRoot::NotRoot);
    hash_parent(&left_hash, &right_hash, key, is_root)
}

pub fn hash(input: &[u8], key: &[u8; KEY_BYTES]) -> [u8; OUT_BYTES] {
    let key_words = words_from_key_bytes(key);
    let hash_words = hash_recurse(input, &key_words, 0, IsRoot::Root);
    bytes_from_state_words(&hash_words)
}

// =======================================================================
// ================== Iterative tree hash implementation =================
// =======================================================================

#[derive(Clone)]
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

    fn maybe_chunk_start_flag(&self) -> Flags {
        // Again, self.count never wraps to the next chunk start.
        let no_blocks_yet = self.count % CHUNK_BYTES as u64 == 0;
        if no_blocks_yet {
            Flags::CHUNK_START
        } else {
            Flags::empty()
        }
    }

    fn append(&mut self, mut input: &[u8]) {
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                debug_assert_eq!(self.buf_len as usize, BLOCK_BYTES);
                let block_words = words_from_msg_bytes(&self.buf);
                let flag = self.maybe_chunk_start_flag();
                compress(
                    &mut self.half_state,
                    &block_words,
                    self.count,
                    BLOCK_BYTES as Word,
                    flag,
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
            let flag = self.maybe_chunk_start_flag();
            compress(
                &mut self.half_state,
                &block_words,
                self.count,
                BLOCK_BYTES as Word,
                flag,
            );
            self.count += BLOCK_BYTES as u64;
            input = &input[BLOCK_BYTES..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
        debug_assert!(self.len() <= CHUNK_BYTES);
    }

    fn finalize(&self, is_root: IsRoot) -> [Word; 8] {
        let mut output = self.half_state;
        let block_words = words_from_msg_bytes(&self.buf);
        compress(
            &mut output,
            &block_words,
            self.count,
            self.buf_len as Word,
            self.maybe_chunk_start_flag() | Flags::CHUNK_END | is_root.flag(),
        );
        output
    }
}

#[derive(Clone)]
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
        // This isn't called during finalization, so this merge is non-root.
        let hash = hash_parent(left_hash, right_hash, &self.key, IsRoot::NotRoot);
        *array_mut_ref!(self.subtree_hash_words, left_start, 8) = hash;
        let new_len = self.subtree_hash_words.len() - 8;
        self.subtree_hash_words.truncate(new_len)
    }

    pub fn append(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_BYTES {
                let chunk_hash = self.chunk.finalize(IsRoot::NotRoot);
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

    pub fn finalize(&self) -> [u8; OUT_BYTES] {
        // If the current chunk is the only chunk, that makes it the root node
        // also. Hash it with the root flag and return that hash. Otherwise, we
        // need to merge all the existing subtrees.
        if self.subtree_hash_words.is_empty() {
            let hash = self.chunk.finalize(IsRoot::Root);
            bytes_from_state_words(&hash)
        } else {
            let mut hash = self.chunk.finalize(IsRoot::NotRoot);
            // Merge that rightmost chunk hash with every hash in the subtree
            // stack, from right (smallest) to left (largest) to produce the
            // final root hash.
            let mut subtrees_remaining = self.num_subtrees();
            for subtree in self.subtree_hash_words.chunks_exact(8).rev() {
                let is_root = if subtrees_remaining == 1 {
                    IsRoot::Root
                } else {
                    IsRoot::NotRoot
                };
                hash = hash_parent(array_ref!(subtree, 0, 8), &hash, &self.key, is_root);
                subtrees_remaining -= 1;
            }
            debug_assert_eq!(subtrees_remaining, 0);
            bytes_from_state_words(&hash)
        }
    }
}

// Implement Debug manually for two reasons. 1) ArrayVec doesn't implement
// Debug for large lengths yet. 2) We don't want to print subtree hashes,
// because they might be secret.
impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hasher {{ count: {} }}", self.chunk.total_count())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_largest_power_of_two_leq() {
        let input_output = &[
            // The zero case is nonsensical, but it does work.
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 8),
            // the largest possible u64
            (0xffffffffffffffff, 0x8000000000000000),
        ];
        for &(input, output) in input_output {
            assert_eq!(
                output,
                largest_power_of_two_leq(input),
                "wrong output for n={}",
                input
            );
        }
    }

    #[test]
    fn test_left_len() {
        let input_output = &[
            (CHUNK_BYTES + 1, CHUNK_BYTES),
            (2 * CHUNK_BYTES - 1, CHUNK_BYTES),
            (2 * CHUNK_BYTES, CHUNK_BYTES),
            (2 * CHUNK_BYTES + 1, 2 * CHUNK_BYTES),
            (4 * CHUNK_BYTES - 1, 2 * CHUNK_BYTES),
            (4 * CHUNK_BYTES, 2 * CHUNK_BYTES),
            (4 * CHUNK_BYTES + 1, 4 * CHUNK_BYTES),
        ];
        for &(input, output) in input_output {
            assert_eq!(left_len(input), output);
        }
    }

    // Interesting input lengths to run tests on.
    pub const TEST_CASES: &[usize] = &[
        0,
        1,
        10,
        CHUNK_BYTES - 1,
        CHUNK_BYTES,
        CHUNK_BYTES + 1,
        2 * CHUNK_BYTES - 1,
        2 * CHUNK_BYTES,
        2 * CHUNK_BYTES + 1,
        3 * CHUNK_BYTES - 1,
        3 * CHUNK_BYTES,
        3 * CHUNK_BYTES + 1,
        4 * CHUNK_BYTES - 1,
        4 * CHUNK_BYTES,
        4 * CHUNK_BYTES + 1,
        8 * CHUNK_BYTES - 1,
        8 * CHUNK_BYTES,
        8 * CHUNK_BYTES + 1,
    ];

    // Paint a byte pattern that won't repeat, so that we don't accidentally
    // miss buffer offset bugs.
    fn paint_test_input(buf: &mut [u8]) {
        let mut offset = 0;
        let mut counter: u32 = 1;
        while offset < buf.len() {
            let bytes = counter.to_le_bytes();
            let take = cmp::min(bytes.len(), buf.len() - offset);
            buf[offset..][..take].copy_from_slice(&bytes[..take]);
            counter += 1;
            offset += take;
        }
    }

    #[test]
    fn test_recursive_incremental_same() {
        let mut input_buf = [0; 65536];
        paint_test_input(&mut input_buf);
        for &case in TEST_CASES {
            let input = &input_buf[..case];
            let key = array_ref!(input_buf, 0, KEY_BYTES);

            let recursive_hash = hash(input, key);

            let mut hasher_all = Hasher::new(key);
            hasher_all.append(input);
            let incremental_hash_all = hasher_all.finalize();
            assert_eq!(recursive_hash, incremental_hash_all);

            let mut hasher_one_at_a_time = Hasher::new(key);
            for &byte in input {
                hasher_one_at_a_time.append(&[byte]);
            }
            let incremental_hash_one_at_a_time = hasher_one_at_a_time.finalize();
            assert_eq!(recursive_hash, incremental_hash_one_at_a_time);
        }
    }

    #[test]
    fn test_zero_bytes() {
        let mut key = [0; KEY_BYTES];
        paint_test_input(&mut key);
        let key_words = words_from_key_bytes(&key);
        let mut state = iv(&key_words);
        let block = [0; BLOCK_BYTES];
        let block_words = words_from_msg_bytes(&block);
        let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
        compress(&mut state, &block_words, 0, 0, flags);
        let expected_hash = bytes_from_state_words(&state);

        assert_eq!(expected_hash, hash(&[], &key));

        let hasher = Hasher::new(&key);
        assert_eq!(expected_hash, hasher.finalize());
    }

    #[test]
    fn test_one_byte() {
        let mut key = [0; KEY_BYTES];
        paint_test_input(&mut key);
        let key_words = words_from_key_bytes(&key);
        let mut state = iv(&key_words);
        let mut block = [0; BLOCK_BYTES];
        block[0] = 9;
        let block_words = words_from_msg_bytes(&block);
        let flags = Flags::CHUNK_START | Flags::CHUNK_END | Flags::ROOT;
        compress(&mut state, &block_words, 0, 1, flags);
        let expected_hash = bytes_from_state_words(&state);

        assert_eq!(expected_hash, hash(&[9], &key));

        let mut hasher = Hasher::new(&key);
        hasher.append(&[9]);
        assert_eq!(expected_hash, hasher.finalize());
    }

    #[test]
    fn test_three_blocks() {
        let mut input = [0; 2 * BLOCK_BYTES + 1];
        paint_test_input(&mut input);
        let mut key = [0; KEY_BYTES];
        paint_test_input(&mut key);
        let key_words = words_from_key_bytes(&key);
        let mut state = iv(&key_words);

        let block0 = array_ref!(input, 0, BLOCK_BYTES);
        let block0_words = words_from_msg_bytes(block0);
        compress(
            &mut state,
            &block0_words,
            0,
            BLOCK_BYTES as Word,
            Flags::CHUNK_START,
        );

        let block1 = array_ref!(input, BLOCK_BYTES, BLOCK_BYTES);
        let block1_words = words_from_msg_bytes(block1);
        compress(
            &mut state,
            &block1_words,
            BLOCK_BYTES as u64,
            BLOCK_BYTES as Word,
            Flags::empty(),
        );

        let mut block2 = [0; BLOCK_BYTES];
        block2[0] = *input.last().unwrap();
        let block2_words = words_from_msg_bytes(&block2);
        compress(
            &mut state,
            &block2_words,
            2 * BLOCK_BYTES as u64,
            1,
            Flags::CHUNK_END | Flags::ROOT,
        );

        let expected_hash = bytes_from_state_words(&state);

        assert_eq!(expected_hash, hash(&input, &key));

        let mut hasher = Hasher::new(&key);
        hasher.append(&input);
        assert_eq!(expected_hash, hasher.finalize());
    }
}
