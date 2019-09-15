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

use arrayref::{array_mut_ref, array_ref, array_refs, mut_array_refs};
use arrayvec::{ArrayString, ArrayVec};
use core::cmp;
use core::fmt;
use platform::Platform;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
mod platform;
mod portable;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41;

#[cfg(test)]
mod test;

type Word = u32;

const WORD_BYTES: usize = core::mem::size_of::<Word>();
const WORD_BITS: usize = 8 * WORD_BYTES;
const MAX_DEPTH: usize = 52; // 2^52 * 4096 = 2^64
const DEFAULT_KEY: &[u8; KEY_BYTES] = &[0; KEY_BYTES];

pub const OUT_BYTES: usize = 8 * WORD_BYTES;
pub const KEY_BYTES: usize = 8 * WORD_BYTES;

// These are pub for tests and benchmarks. Callers don't need them.
#[doc(hidden)]
pub const BLOCK_BYTES: usize = 16 * WORD_BYTES;
#[doc(hidden)]
pub const CHUNK_BYTES: usize = 4096;
#[doc(hidden)]
pub use platform::MAX_SIMD_DEGREE;

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
        const KDF = 16;
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

impl PartialEq<[u8; OUT_BYTES]> for Hash {
    fn eq(&self, other: &[u8; OUT_BYTES]) -> bool {
        constant_time_eq::constant_time_eq(&self.0[..], other)
    }
}

impl Eq for Hash {}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

/// A very simple interface to the XOF, as a proof of concept. A more
/// fleshed-out implementation would implement `std::io::{Read, Seek}` and
/// would use SIMD parallelism internally.
#[derive(Clone)]
pub struct Output {
    state: [Word; 8],
    block: [u8; BLOCK_BYTES],
    block_len: u8,
    offset: u64,
    flags: Flags,
    platform: Platform,
}

impl Output {
    pub fn read(&mut self) -> [u8; OUT_BYTES] {
        let mut state_copy = self.state;
        self.platform.compress(
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

// Derive an empty Debug impl, because the contents might be secret.
impl fmt::Debug for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Output {{ ... }}")
    }
}

// =======================================================================
// ================== Recursive tree hash implementation =================
// =======================================================================

// This isn't as fast as hash_chunks_parallel when multiple chunks are
// available, but it handles partial chunks and root/XOF finalization.
//
// Note that when we use this function internally, we might build Output
// objects without the root flag and with non-zero offsets. But Output objects
// returned publicly are always root finalizations starting at offset zero.
// That's because the root is either the only chunk (which is always going to
// sit at offset zero) or a parent node (which is offset zero by definition).
fn hash_one_chunk(
    chunk: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
    is_root: IsRoot,
    platform: Platform,
) -> Output {
    if let IsRoot::Root = is_root {
        debug_assert_eq!(offset, 0, "root chunk must be offset 0");
    }
    let mut state = iv(key);
    let mut block_flags = flags | Flags::CHUNK_START;
    let mut block_offset = 0;
    while block_offset + BLOCK_BYTES <= chunk.len() {
        if block_offset + BLOCK_BYTES == chunk.len() {
            return Output {
                state,
                block: *array_ref!(chunk, block_offset, BLOCK_BYTES),
                block_len: BLOCK_BYTES as u8,
                offset,
                flags: block_flags | Flags::CHUNK_END | is_root.flag(),
                platform,
            };
        }
        platform.compress(
            &mut state,
            array_ref!(chunk, block_offset, BLOCK_BYTES),
            BLOCK_BYTES as Word,
            offset,
            block_flags.bits(),
        );
        block_offset += BLOCK_BYTES;
        block_flags = flags;
    }

    // There's a partial block left over (or the whole chunk was empty, which
    // only happens when the whole message is empty). Pad the last block with
    // zero bytes return it as an Output.
    let block_len = chunk.len() - block_offset;
    let mut last_block = [0; BLOCK_BYTES];
    last_block[..block_len].copy_from_slice(&chunk[block_offset..]);
    Output {
        state,
        block: last_block,
        block_len: block_len as u8,
        offset,
        // Note that block_flags may still contain CHUNK_START.
        flags: block_flags | Flags::CHUNK_END | is_root.flag(),
        platform,
    }
}

// Use SIMD parallelism to hash multiple full chunks at the same time on a
// single thread. Returns the number of chunks hashed. These chunks are never
// the root and never empty.
fn hash_chunks_parallel(
    input: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
    platform: Platform,
    out: &mut [u8],
) -> usize {
    debug_assert!(!input.is_empty(), "empty chunks below the root");
    debug_assert!(input.len() <= MAX_SIMD_DEGREE * CHUNK_BYTES);
    debug_assert_eq!(offset % CHUNK_BYTES as u64, 0, "invalid offset");

    let mut chunks_exact = input.chunks_exact(CHUNK_BYTES);
    let mut chunks_array = ArrayVec::<[&[u8; CHUNK_BYTES]; MAX_SIMD_DEGREE]>::new();
    for chunk in &mut chunks_exact {
        chunks_array.push(array_ref!(chunk, 0, CHUNK_BYTES));
    }
    platform.hash_many_chunks(
        &chunks_array,
        key,
        offset,
        flags.bits(),
        Flags::CHUNK_START.bits(),
        Flags::CHUNK_END.bits(),
        out,
    );

    // Handle the remaining partial chunk, if there is one. Note that the empty
    // chunk (meaning the empty message) is a different codepath.
    let chunks_so_far = chunks_array.len();
    if !chunks_exact.remainder().is_empty() {
        let mut output = hash_one_chunk(
            chunks_exact.remainder(),
            key,
            offset + chunks_so_far as u64 * CHUNK_BYTES as u64,
            flags,
            IsRoot::NotRoot,
            platform,
        );
        *array_mut_ref!(out, chunks_so_far * OUT_BYTES, OUT_BYTES) = output.read();
        chunks_so_far + 1
    } else {
        chunks_so_far
    }
}

// Use SIMD parallelism to hash multiple concatenated parents at the same time
// on a single thread. If there's an odd child left over, concatenate it to the
// end of the output. Return the number of outputs written. These parents are
// never the root.
fn hash_parents_parallel(
    child_hashes: &[u8],
    key: &[Word; 8],
    flags: Flags,
    platform: Platform,
    out: &mut [u8],
) -> usize {
    debug_assert_eq!(child_hashes.len() % OUT_BYTES, 0, "wacky hash bytes");
    let num_children = child_hashes.len() / OUT_BYTES;
    debug_assert!(num_children >= 2, "not enough children");
    debug_assert!(num_children <= 2 * MAX_SIMD_DEGREE, "too many");

    let mut parents_exact = child_hashes.chunks_exact(BLOCK_BYTES);
    let mut parents_array = ArrayVec::<[&[u8; BLOCK_BYTES]; MAX_SIMD_DEGREE]>::new();
    for parent in &mut parents_exact {
        parents_array.push(array_ref!(parent, 0, BLOCK_BYTES));
    }
    let parent_flags = flags | Flags::PARENT;
    platform.hash_many_parents(&parents_array, key, parent_flags.bits(), out);

    // If there's an odd child left over, it becomes an output.
    let parents_so_far = parents_array.len();
    if !parents_exact.remainder().is_empty() {
        out[parents_so_far * OUT_BYTES..][..OUT_BYTES].copy_from_slice(parents_exact.remainder());
        parents_so_far + 1
    } else {
        parents_so_far
    }
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

fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "rayon")]
    return rayon::join(oper_a, oper_b);
    #[cfg(not(feature = "rayon"))]
    return (oper_a(), oper_b());
}

fn hash_recurse(
    input: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
    platform: Platform,
    out: &mut [u8],
) -> usize {
    if input.len() <= platform.simd_degree() * CHUNK_BYTES {
        return hash_chunks_parallel(input, key, offset, flags, platform, out);
    }

    // With more than simd_degree chunks, we need to recurse. Divide the input
    // up so that the largest possible power of two chunks goes on the left,
    // and at least one byte goes on the right. The simd_degree is assumed to
    // be a power of two also, which means we will get that many outputs from
    // the left, and we split the out slice accordingly. All the outputs then
    // wind up adjacent to each other for the call to hash_parents_parallel.
    let (left, right) = input.split_at(left_len(input.len()));
    let right_off = offset + left.len() as u64;
    debug_assert_eq!(platform.simd_degree().count_ones(), 1, "power of two");
    let mut children_array = [0; 2 * MAX_SIMD_DEGREE * OUT_BYTES];
    let (left_out, right_out) = children_array.split_at_mut(platform.simd_degree() * OUT_BYTES);

    let (left_n, right_n) = join(
        || hash_recurse(left, key, offset, flags, platform, left_out),
        || hash_recurse(right, key, right_off, flags, platform, right_out),
    );

    debug_assert_eq!(left_n, platform.simd_degree(), "unexpected left children");
    let num_children = left_n + right_n;
    hash_parents_parallel(
        &children_array[..num_children * OUT_BYTES],
        key,
        flags,
        platform,
        out,
    )
}

fn condense_root(
    mut children: &mut [u8],
    key: &[Word; 8],
    flags: Flags,
    platform: Platform,
) -> Output {
    debug_assert_eq!(children.len() % OUT_BYTES, 0);
    debug_assert!(children.len() >= BLOCK_BYTES);
    let mut out_array = [0; MAX_SIMD_DEGREE * OUT_BYTES / 2];
    while children.len() > BLOCK_BYTES {
        let out_n = hash_parents_parallel(children, key, flags, platform, &mut out_array);
        children[..out_n * OUT_BYTES].copy_from_slice(&out_array[..out_n * OUT_BYTES]);
        children = &mut children[..out_n * OUT_BYTES];
    }
    Output {
        state: iv(key),
        block: *array_ref!(children, 0, BLOCK_BYTES),
        block_len: BLOCK_BYTES as u8,
        offset: 0,
        flags: flags | Flags::PARENT | Flags::ROOT,
        platform,
    }
}

fn hash_keyed_flags_xof(input: &[u8], key: &[u8; KEY_BYTES], flags: Flags) -> Output {
    let platform = Platform::detect();
    let key_words = words_from_key_bytes(key);
    if input.len() <= CHUNK_BYTES {
        return hash_one_chunk(input, &key_words, 0, flags, IsRoot::Root, platform);
    }
    let mut children_array = [0; MAX_SIMD_DEGREE * OUT_BYTES];
    let num_children = if input.len() <= platform.simd_degree() * CHUNK_BYTES {
        // We don't call hash_recurse when there are <= simd_degree children,
        // because in that cases it might need to root/XOF finalize, and we
        // want to do that here instead.
        hash_chunks_parallel(input, &key_words, 0, flags, platform, &mut children_array)
    } else {
        hash_recurse(input, &key_words, 0, flags, platform, &mut children_array)
    };
    condense_root(
        &mut children_array[..num_children * OUT_BYTES],
        &key_words,
        flags,
        platform,
    )
}

pub fn hash_keyed_xof(input: &[u8], key: &[u8; KEY_BYTES]) -> Output {
    hash_keyed_flags_xof(input, key, Flags::empty())
}

pub fn hash_xof(input: &[u8]) -> Output {
    hash_keyed_flags_xof(input, DEFAULT_KEY, Flags::empty())
}

pub fn hash_keyed(input: &[u8], key: &[u8; KEY_BYTES]) -> Hash {
    hash_keyed_flags_xof(input, key, Flags::empty())
        .read()
        .into()
}

pub fn hash(input: &[u8]) -> Hash {
    hash_keyed_flags_xof(input, DEFAULT_KEY, Flags::empty())
        .read()
        .into()
}

pub fn kdf(key: &[u8; KEY_BYTES], context: &[u8]) -> [u8; KEY_BYTES] {
    hash_keyed_flags_xof(context, key, Flags::KDF).read()
}

pub fn kdf_xof(key: &[u8; KEY_BYTES], context: &[u8]) -> Output {
    hash_keyed_flags_xof(context, key, Flags::KDF)
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

    fn append(&mut self, mut input: &[u8], flags: Flags, platform: Platform) {
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                debug_assert_eq!(self.buf_len as usize, BLOCK_BYTES);
                let block_flags = flags | self.maybe_chunk_start_flag();
                platform.compress(
                    &mut self.state,
                    &self.buf,
                    BLOCK_BYTES as Word,
                    self.offset,
                    block_flags.bits(),
                );
                self.count += BLOCK_BYTES as u16;
                self.buf_len = 0;
                self.buf = [0; BLOCK_BYTES];
            }
        }

        while input.len() > BLOCK_BYTES {
            debug_assert_eq!(self.buf_len, 0);
            let block_flags = flags | self.maybe_chunk_start_flag();
            platform.compress(
                &mut self.state,
                array_ref!(input, 0, BLOCK_BYTES),
                BLOCK_BYTES as Word,
                self.offset,
                block_flags.bits(),
            );
            self.count += BLOCK_BYTES as u16;
            input = &input[BLOCK_BYTES..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
        debug_assert!(self.len() <= CHUNK_BYTES);
    }

    // Note that similar to hash_one_chunk, when we use this function
    // internally, we might build Output objects without the root flag and with
    // non-zero offsets. But Output objects returned publicly are always root
    // finalizations starting at offset zero.
    fn finalize(&self, flags: Flags, is_root: IsRoot, platform: Platform) -> Output {
        if let IsRoot::Root = is_root {
            // Root finalization starts with offset 0 to support the XOF, so
            // it's convenient that only the chunk at offset 0 can be the root.
            debug_assert_eq!(self.offset, 0);
        }
        Output {
            state: self.state,
            block: self.buf,
            block_len: self.buf_len,
            offset: self.offset,
            flags: flags | self.maybe_chunk_start_flag() | Flags::CHUNK_END | is_root.flag(),
            platform,
        }
    }
}

// This isn't as fast as hash_parents_parallel when multiple parents are
// available, but it handles root/XOF finalization.
//
// Note that similar to hash_one_chunk, when we use this function internally,
// we might build Output objects without the root flag. But Output objects
// returned publicly are always root finalizations.
fn hash_one_parent(
    left_child: &[u8; OUT_BYTES],
    right_child: &[u8; OUT_BYTES],
    key: &[Word; 8],
    flags: Flags,
    is_root: IsRoot,
    platform: Platform,
) -> Output {
    let mut block = [0; BLOCK_BYTES];
    let refs = mut_array_refs!(&mut block, OUT_BYTES, OUT_BYTES);
    *refs.0 = *left_child;
    *refs.1 = *right_child;
    Output {
        state: iv(key),
        block,
        block_len: BLOCK_BYTES as u8,
        offset: 0,
        flags: flags | Flags::PARENT | is_root.flag(),
        platform,
    }
}

#[derive(Clone)]
pub struct Hasher {
    subtree_hashes: ArrayVec<[[u8; OUT_BYTES]; MAX_DEPTH]>,
    chunk: ChunkState,
    key: [Word; 8],
    flags: Flags,
    platform: Platform,
}

impl Hasher {
    pub fn new() -> Self {
        Self::new_keyed(&[0; KEY_BYTES])
    }

    pub fn new_keyed(key: &[u8; KEY_BYTES]) -> Self {
        Self::new_keyed_flags(key, Flags::empty())
    }

    fn new_keyed_flags(key_bytes: &[u8; KEY_BYTES], flags: Flags) -> Self {
        let key = words_from_key_bytes(key_bytes);
        Self {
            subtree_hashes: ArrayVec::new(),
            chunk: ChunkState::new(&key, 0),
            key,
            flags,
            platform: Platform::detect(),
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
        let hash = hash_one_parent(
            &self.subtree_hashes[num_subtrees - 2],
            &self.subtree_hashes[num_subtrees - 1],
            &self.key,
            self.flags,
            IsRoot::NotRoot,
            self.platform,
        )
        .read();
        self.subtree_hashes[num_subtrees - 2] = hash;
        self.subtree_hashes.truncate(num_subtrees - 1)
    }

    fn push_chunk_hash(&mut self, hash: &[u8; OUT_BYTES], offset: u64) {
        // Do subtree merging *before* pushing the new hash. That way we know
        // that the merged parent node isn't the root.
        while self.needs_merge(offset) {
            self.merge_parent();
        }
        self.subtree_hashes.push(*hash);
    }

    pub fn append(&mut self, mut input: &[u8]) -> &mut Self {
        // When we have whole chunks coming in, hash_chunks_parallel gives the
        // best performance. However, we have to be careful with the very first
        // chunk. If we don't have more input yet, then we don't know whether
        // it's the root, and we have to keep it in the ChunkState until we
        // find out. Also, if we have any partial chunk bytes in the ChunkState
        // already, we need to finish it.
        let is_first_chunk = self.chunk.offset == 0;
        let maybe_root = is_first_chunk && input.len() == CHUNK_BYTES;
        if maybe_root || self.chunk.len() > 0 {
            let want = CHUNK_BYTES - self.chunk.len();
            let take = cmp::min(want, input.len());
            self.chunk.append(&input[..take], self.flags, self.platform);
            input = &input[take..];
            if !input.is_empty() {
                // We've filled the current chunk, and there's more input
                // coming, so we know it's not the root and we can finalize it.
                // Then we'll proceed to hashing whole chunks below.
                debug_assert_eq!(self.chunk.len(), CHUNK_BYTES);
                let chunk_hash = self
                    .chunk
                    .finalize(self.flags, IsRoot::NotRoot, self.platform)
                    .read();
                self.push_chunk_hash(&chunk_hash, self.chunk.offset);
                self.chunk = ChunkState::new(&self.key, self.chunk.offset + CHUNK_BYTES as u64);
            } else {
                return self;
            }
        }

        // At this point, the ChunkState is clear. Hash all the full chunks
        // that we can, using SIMD parallelism. But note that this still
        // single-threaded. (Only the recursive hash functions are
        // multi-threaded.)
        let mut chunks_exact = input.chunks_exact(CHUNK_BYTES);
        let mut fused_chunks = chunks_exact
            .by_ref()
            .map(|chunk| array_ref!(chunk, 0, CHUNK_BYTES))
            .fuse();
        let mut chunks_array = ArrayVec::<[&[u8; CHUNK_BYTES]; MAX_SIMD_DEGREE]>::new();
        let mut out_array = [0; MAX_SIMD_DEGREE * OUT_BYTES];
        debug_assert_eq!(self.chunk.len(), 0);
        loop {
            chunks_array.extend(fused_chunks.by_ref().take(MAX_SIMD_DEGREE));
            if chunks_array.is_empty() {
                // We've exhausted the whole chunks. Add any remaining input to
                // the ChunkState below.
                break;
            }
            self.platform.hash_many_chunks(
                &chunks_array,
                &self.key,
                self.chunk.offset,
                self.flags.bits(),
                Flags::CHUNK_START.bits(),
                Flags::CHUNK_END.bits(),
                &mut out_array,
            );
            for chunk_hash in out_array.chunks_exact(OUT_BYTES).take(chunks_array.len()) {
                self.push_chunk_hash(array_ref!(chunk_hash, 0, OUT_BYTES), self.chunk.offset);
                // Move the ChunkState's offset forward after each hash. This
                // is safe because it's clear, and it leaves it in the right
                // position for any remainder bytes below or subsequent calls
                // to append.
                self.chunk.offset += CHUNK_BYTES as u64;
            }
            chunks_array.clear();
        }

        // The loop above hashed all the whole chunks it could. If there are
        // any bytes left, add them to the ChunkState. The ChunkState's offset
        // may have been updated above.
        debug_assert_eq!(self.chunk.len(), 0);
        debug_assert!(chunks_exact.remainder().len() < CHUNK_BYTES);
        if !chunks_exact.remainder().is_empty() {
            // Working ahead: The subtree_hashes stack might contain subtrees
            // that need to be merged. Normally these would be merged by the
            // next call to push_subtree. However, if we go ahead and take care
            // of it now (since we see there's more input), that will simplify
            // finalization later.
            while self.needs_merge(self.chunk.offset) {
                self.merge_parent();
            }
            self.chunk
                .append(chunks_exact.remainder(), self.flags, self.platform);
        }
        self
    }

    pub fn finalize(&self) -> Hash {
        self.finalize_xof().read().into()
    }

    pub fn finalize_xof(&self) -> Output {
        // If the current chunk is the only chunk, that makes it the root node
        // also. Convert it directly into an Output. Otherwise, we need to
        // merge subtrees below.
        if self.subtree_hashes.is_empty() {
            return self.chunk.finalize(self.flags, IsRoot::Root, self.platform);
        }

        // If there are any bytes in the ChunkState, finalize that chunk and
        // merge it with everything in the subtree stack. In that case, the
        // work we did at the end of append above guarantees that the stack
        // doesn't contain any unmerged subtrees that need to be merged first.
        // (This is important, because if there were two chunk hashes sitting
        // on top of the stack, they would need to merge with each other, and
        // merging a new chunk hash into them would be incorrect.)
        //
        // If there are no bytes in the ChunkState, we'll merge what's already
        // in the stack. In this case it's fine if there are unmerged chunks on
        // top, because we'll merge them with each other.
        let mut working_hash;
        let mut next_subtree_index;
        if self.chunk.len() > 0 {
            debug_assert!(!self.needs_merge(self.chunk.offset));
            working_hash = self
                .chunk
                .finalize(self.flags, IsRoot::NotRoot, self.platform)
                .read();
            next_subtree_index = self.subtree_hashes.len() - 1;
        } else {
            debug_assert!(self.subtree_hashes.len() >= 2);
            working_hash = *self.subtree_hashes.last().unwrap();
            next_subtree_index = self.subtree_hashes.len() - 2;
        }
        loop {
            let is_root = if next_subtree_index == 0 {
                IsRoot::Root
            } else {
                IsRoot::NotRoot
            };
            let mut output = hash_one_parent(
                &self.subtree_hashes[next_subtree_index],
                &working_hash,
                &self.key,
                self.flags,
                is_root,
                self.platform,
            );
            if next_subtree_index == 0 {
                return output;
            }
            working_hash = output.read();
            next_subtree_index -= 1;
        }
    }
}

// Derive an empty Debug impl, because the contents might be secret.
impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hasher {{ ... }}")
    }
}
