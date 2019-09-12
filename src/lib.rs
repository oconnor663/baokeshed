// Design notes:
//
// The CHUNK_END and PARENT flags aren't strictly necessary, the former because
// the keyed root node makes it impossible to extend a chunk hash without the
// key, and the latter because CHUNK_START already effectively domain-separates
// chunks from parent nodes. But do we want to have to think about what happens
// when someone length-extends a non-root keyed chunk, or what happens when
// someone "smuggles" a chunk CV into a parent IV via the key? It doesn't cost
// much to be conservative here.
//
// Rather than incrementing the count for each chunk, we can keep it set to the
// chunk offset for all the blocks of that chunk, and start calling it the
// "offset". That preserves the "no dangerous caching optimizations" behavior,
// and it leads to two nice simplifications: 1) Setting the offset back to zero
// when we finalize a root chunk is no longer a special case. It's already
// zero. 2) The SIMD chunk compression loop doesn't need to increment the
// offset. The offset is split into two vectors of 32-bit words at that point,
// and leaving those words constant lets us delete the instructions that were
// doing wrapping addition.

use arrayref::{array_ref, array_refs, mut_array_refs};
use arrayvec::{ArrayString, ArrayVec};
use core::cmp;
use core::fmt;

mod portable;

#[cfg(test)]
mod test;

type Word = u32;

const WORD_BYTES: usize = core::mem::size_of::<Word>();
const WORD_BITS: usize = 8 * WORD_BYTES;
const MAX_DEPTH: usize = 52; // 2^52 * 4096 = 2^64
pub const OUT_BYTES: usize = 8 * WORD_BYTES;
pub const KEY_BYTES: usize = 8 * WORD_BYTES;
#[doc(hidden)] // for benchmarks
pub const BLOCK_BYTES: usize = 16 * WORD_BYTES;
#[doc(hidden)] // for benchmarks
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

fn words_from_key_bytes(bytes: &[u8; KEY_BYTES]) -> [Word; 8] {
    // Parse the message bytes as little endian words.
    let refs = array_refs!(bytes, 4, 4, 4, 4, 4, 4, 4, 4);
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
        let refs = mut_array_refs!(&mut bytes, 4, 4, 4, 4, 4, 4, 4, 4);
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

fn hash_from_state_words(words: &[Word; 8]) -> Hash {
    bytes_from_state_words(words).into()
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

#[derive(Clone, Copy)]
pub struct Hash([u8; OUT_BYTES]);

impl Hash {
    pub fn as_bytes(&self) -> &[u8; OUT_BYTES] {
        &self.0
    }

    pub fn to_hex(&self) -> ArrayString<[u8; 2 * OUT_BYTES]> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.0.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }
}

impl From<[u8; OUT_BYTES]> for Hash {
    fn from(bytes: [u8; OUT_BYTES]) -> Self {
        Self(bytes)
    }
}

impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.0[..], &other.0[..])
    }
}

impl PartialEq<[u8]> for Hash {
    fn eq(&self, other: &[u8]) -> bool {
        constant_time_eq::constant_time_eq(&self.0[..], other)
    }
}

impl Eq for Hash {}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

#[derive(Clone)]
pub struct Output {
    state: [Word; 8],
    block: [u8; BLOCK_BYTES],
    block_len: u8,
    flags: Flags,
    offset: u64,
}

impl Output {
    pub fn read(&mut self) -> [u8; OUT_BYTES] {
        debug_assert!(self.flags.contains(Flags::ROOT));
        let mut state_copy = self.state;
        portable::compress(
            &mut state_copy,
            &self.block,
            self.block_len as Word,
            self.offset,
            self.flags.bits(),
        );
        self.offset += OUT_BYTES as u64;
        bytes_from_state_words(&state_copy)
    }
}

// =======================================================================
// ================== Recursive tree hash implementation =================
// =======================================================================

fn hash_chunk(mut chunk: &[u8], key: &[Word; 8], offset: u64, is_root: IsRoot) -> [Word; 8] {
    debug_assert!(chunk.len() <= CHUNK_BYTES);
    debug_assert_eq!(offset % CHUNK_BYTES as u64, 0);
    if let IsRoot::Root = is_root {
        // Root finalization starts with offset 0 to support the XOF, so it's
        // convenient that only the chunk at offset 0 can be the root.
        debug_assert_eq!(offset, 0);
    }
    let mut state = iv(key);
    let mut maybe_start_flag = Flags::CHUNK_START;
    while chunk.len() > BLOCK_BYTES {
        portable::compress(
            &mut state,
            array_ref!(chunk, 0, BLOCK_BYTES),
            BLOCK_BYTES as Word,
            offset,
            maybe_start_flag.bits(),
        );
        chunk = &chunk[BLOCK_BYTES..];
        maybe_start_flag = Flags::empty();
    }
    let mut last_block = [0; BLOCK_BYTES];
    last_block[..chunk.len()].copy_from_slice(chunk);
    let flags = maybe_start_flag | Flags::CHUNK_END | is_root.flag();
    portable::compress(
        &mut state,
        &last_block,
        chunk.len() as Word,
        offset,
        flags.bits(),
    );
    state
}

fn concat_parent(left_hash: &[Word; 8], right_hash: &[Word; 8]) -> [u8; BLOCK_BYTES] {
    let mut block = [0; BLOCK_BYTES];
    let refs = mut_array_refs!(&mut block, OUT_BYTES, OUT_BYTES);
    *refs.0 = bytes_from_state_words(left_hash);
    *refs.1 = bytes_from_state_words(right_hash);
    block
}

fn hash_parent(
    left_hash: &[Word; 8],
    right_hash: &[Word; 8],
    key: &[Word; 8],
    is_root: IsRoot,
) -> [Word; 8] {
    let mut state = iv(key);
    let flags = Flags::PARENT | is_root.flag();
    portable::compress(
        &mut state,
        &concat_parent(left_hash, right_hash),
        BLOCK_BYTES as Word,
        0, // Note that parents always use offset zero.
        flags.bits(),
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

fn hash_recurse(input: &[u8], key: &[Word; 8], offset: u64, is_root: IsRoot) -> [Word; 8] {
    if input.len() <= CHUNK_BYTES {
        return hash_chunk(input, key, offset, is_root);
    }
    let (left, right) = input.split_at(left_len(input.len()));
    let right_offset = offset + left.len() as u64;

    #[cfg(feature = "rayon")]
    let children = rayon::join(
        || hash_recurse(left, key, offset, IsRoot::NotRoot),
        || hash_recurse(right, key, right_offset, IsRoot::NotRoot),
    );
    #[cfg(not(feature = "rayon"))]
    let children = (
        hash_recurse(left, key, offset, IsRoot::NotRoot),
        hash_recurse(right, key, right_offset, IsRoot::NotRoot),
    );

    hash_parent(&children.0, &children.1, key, is_root)
}

pub fn hash_keyed(input: &[u8], key: &[u8; KEY_BYTES]) -> Hash {
    let key_words = words_from_key_bytes(key);
    let hash_words = hash_recurse(input, &key_words, 0, IsRoot::Root);
    hash_from_state_words(&hash_words)
}

pub fn hash(input: &[u8]) -> Hash {
    hash_keyed(input, &[0; KEY_BYTES])
}

// =======================================================================
// ================== Iterative tree hash implementation =================
// =======================================================================

#[derive(Clone)]
struct ChunkState {
    state: [Word; 8],
    offset: u64,
    count: u16,
    buf: [u8; BLOCK_BYTES],
    buf_len: u8,
}

impl ChunkState {
    fn new(key: &[Word; 8], offset: u64) -> Self {
        debug_assert_eq!(offset % CHUNK_BYTES as u64, 0);
        Self {
            state: iv(key),
            offset,
            count: 0,
            buf: [0; BLOCK_BYTES],
            buf_len: 0,
        }
    }

    // The length of the current chunk, including buffered bytes.
    fn len(&self) -> usize {
        self.count as usize + self.buf_len as usize
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
        if self.count == 0 {
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
                let flags = self.maybe_chunk_start_flag();
                portable::compress(
                    &mut self.state,
                    &self.buf,
                    BLOCK_BYTES as Word,
                    self.offset,
                    flags.bits(),
                );
                self.count += BLOCK_BYTES as u16;
                self.buf_len = 0;
                self.buf = [0; BLOCK_BYTES];
            }
        }

        while input.len() > BLOCK_BYTES {
            debug_assert_eq!(self.buf_len, 0);
            let flags = self.maybe_chunk_start_flag();
            portable::compress(
                &mut self.state,
                array_ref!(input, 0, BLOCK_BYTES),
                BLOCK_BYTES as Word,
                self.offset,
                flags.bits(),
            );
            self.count += BLOCK_BYTES as u16;
            input = &input[BLOCK_BYTES..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
        debug_assert!(self.len() <= CHUNK_BYTES);
    }

    fn finalize(&self, is_root: IsRoot) -> [Word; 8] {
        if let IsRoot::Root = is_root {
            // Root finalization starts with offset 0 to support the XOF, so
            // it's convenient that only the chunk at offset 0 can be the root.
            debug_assert_eq!(self.offset, 0);
        }
        let mut output = self.state;
        let flags = self.maybe_chunk_start_flag() | Flags::CHUNK_END | is_root.flag();
        portable::compress(
            &mut output,
            &self.buf,
            self.buf_len as Word,
            self.offset,
            flags.bits(),
        );
        output
    }

    // IsRoot::Root is implied here.
    fn finalize_xof(&self) -> Output {
        debug_assert_eq!(self.offset, 0);
        Output {
            state: self.state,
            block: self.buf,
            block_len: self.buf_len,
            flags: self.maybe_chunk_start_flag() | Flags::CHUNK_END | Flags::ROOT,
            offset: 0,
        }
    }
}

#[derive(Clone)]
pub struct Hasher {
    key: [Word; 8],
    chunk: ChunkState,
    subtree_hashes: ArrayVec<[[Word; 8]; MAX_DEPTH]>,
}

impl Hasher {
    pub fn new() -> Self {
        Self::new_keyed(&[0; KEY_BYTES])
    }

    pub fn new_keyed(key_bytes: &[u8; KEY_BYTES]) -> Self {
        let key = words_from_key_bytes(key_bytes);
        Self {
            key,
            chunk: ChunkState::new(&key, 0),
            subtree_hashes: ArrayVec::new(),
        }
    }

    // We keep subtree hashes in the subtree_hashes array without storing
    // subtree sizes anywhere, and we use this cute trick to figure out when we
    // should merge them. Because every subtree (prior to the finalization
    // step) is a power of two times the chunk size, adding a new subtree to
    // the right/small end is a lot like adding a 1 to a binary number, and
    // merging subtrees is like propagating the carry bit. Each carry
    // represents a place where two subtrees need to be merged, and the final
    // number of 1 bits is the same as the final number of subtrees.
    fn needs_merge(&self, total_bytes: u64) -> bool {
        let total_chunks = total_bytes / CHUNK_BYTES as u64;
        self.subtree_hashes.len() > total_chunks.count_ones() as usize
    }

    // Take two subtree hashes off the end of the stack, hash them into a
    // parent node, and put that hash back on the stack.
    fn merge_parent(&mut self) {
        let num_subtrees = self.subtree_hashes.len();
        debug_assert!(num_subtrees >= 2);
        let left_hash = &self.subtree_hashes[num_subtrees - 2];
        let right_hash = &self.subtree_hashes[num_subtrees - 1];
        // This isn't called during finalization, so this merge is non-root.
        let hash = hash_parent(left_hash, right_hash, &self.key, IsRoot::NotRoot);
        self.subtree_hashes[num_subtrees - 2] = hash;
        self.subtree_hashes.truncate(num_subtrees - 1)
    }

    pub fn append(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            // More input coming means that we can finish this chunk and merge
            // parents without doing root finalization.
            if self.chunk.len() == CHUNK_BYTES {
                let chunk_hash = self.chunk.finalize(IsRoot::NotRoot);
                self.subtree_hashes.push(chunk_hash);
                let total_bytes = self.chunk.offset + CHUNK_BYTES as u64;
                while self.needs_merge(total_bytes) {
                    self.merge_parent();
                }
                self.chunk = ChunkState::new(&self.key, total_bytes);
            }
            let want = CHUNK_BYTES - self.chunk.len();
            let take = cmp::min(want, input.len());
            self.chunk.append(&input[..take]);
            input = &input[take..];
        }
    }

    pub fn finalize(&self) -> Hash {
        self.finalize_xof().read().into()
    }

    pub fn finalize_xof(&self) -> Output {
        // If the current chunk is the only chunk, that makes it the root node
        // also. Convert it directly into an Output. Otherwise, we need to
        // merge subtrees.
        if self.subtree_hashes.is_empty() {
            self.chunk.finalize_xof()
        } else {
            let mut hash = self.chunk.finalize(IsRoot::NotRoot);
            // Merge that rightmost chunk hash with each successive subtree,
            // from right (smallest) to left (largest) until there's only one
            // subtree remaining. Then convert the final parent node into an
            // Output.
            for subtree_index in (1..self.subtree_hashes.len()).rev() {
                let subtree = &self.subtree_hashes[subtree_index];
                hash = hash_parent(subtree, &hash, &self.key, IsRoot::NotRoot);
            }
            Output {
                state: iv(&self.key),
                block: concat_parent(&self.subtree_hashes[0], &hash),
                block_len: BLOCK_BYTES as u8,
                flags: Flags::PARENT | Flags::ROOT,
                offset: 0,
            }
        }
    }
}

// We don't want to print subtree hashes or other intermediate words, because
// they might be secret, and because they might enable length extension. So we
// implement Debug manually.
impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hasher {{ ... }}")
    }
}
