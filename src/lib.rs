//! There are three domain-separated variants of the hash function:
//!
//! - `hash(input)`
//! - `keyed_hash(input, key)`
//! - `derive_key(key, context)`
//!
//! All three are intended to be collectively collision resistant. The
//! incremental `Hasher` type provides three equivalent constructors.
//!
//! The standalone functions in this module use both SIMD and
//! [Rayon](https://github.com/rayon-rs/rayon)-based multithreading. The
//! streaming [`Hasher`](struct.Hasher.html) implementation uses SIMD but not
//! threads.
//!
//! Encoding and streaming/seeking verification are not yet implemented.
//!
//! - [GitHub repo](https://github.com/oconnor663/baokeshed)

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
use platform::{Platform, MAX_SIMD_DEGREE_OR_2};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(feature = "c_portable")]
pub mod c;
pub mod copy;
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

const ZERO_KEY: &[u8; KEY_LEN] = &[0; KEY_LEN];

/// The default number of bytes in a hash, 32.
pub const OUT_LEN: usize = 8 * WORD_BYTES;

/// The number of bytes in a key, 32.
pub const KEY_LEN: usize = 8 * WORD_BYTES;

// These are pub for tests and benchmarks. Callers don't need them.
#[doc(hidden)]
pub const BLOCK_LEN: usize = 16 * WORD_BYTES;
#[doc(hidden)]
pub const CHUNK_LEN: usize = 4096;
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

// These are the internal flags that we use to domain separate root/non-root,
// chunk/parent, and chunk beginning/middle/end. These get set at the high end
// of the block flags word in the compression function, so their values start
// high and go down.
bitflags::bitflags! {
    struct Flags: u8 {
        const ROOT = 1 << 0;
        const PARENT = 1 << 1;
        const CHUNK_END = 1 << 2;
        const CHUNK_START = 1 << 3;
        const KEYED_HASH = 1 << 4;
        const DERIVE_KEY = 1 << 5;
    }
}

#[derive(Clone, Copy, Debug)]
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

fn words_from_key_bytes(bytes: &[u8; KEY_LEN]) -> [Word; 8] {
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

fn bytes_from_state_words(words: &[Word; 8]) -> [u8; OUT_LEN] {
    let mut bytes = [0; OUT_LEN];
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

fn offset_low(offset: u64) -> Word {
    offset as Word
}

fn offset_high(offset: u64) -> Word {
    (offset >> WORD_BITS) as Word
}

/// A hash output of the default size, providing constant-time equality.
#[derive(Clone, Copy)]
pub struct Hash([u8; OUT_LEN]);

impl Hash {
    pub fn as_bytes(&self) -> &[u8; OUT_LEN] {
        &self.0
    }

    pub fn to_hex(&self) -> ArrayString<[u8; 2 * OUT_LEN]> {
        let mut s = ArrayString::new();
        let table = b"0123456789abcdef";
        for &b in self.0.iter() {
            s.push(table[(b >> 4) as usize] as char);
            s.push(table[(b & 0xf) as usize] as char);
        }
        s
    }
}

impl From<[u8; OUT_LEN]> for Hash {
    fn from(bytes: [u8; OUT_LEN]) -> Self {
        Self(bytes)
    }
}

impl From<Hash> for [u8; OUT_LEN] {
    fn from(hash: Hash) -> Self {
        hash.0
    }
}

/// This implementation is constant-time.
impl PartialEq for Hash {
    fn eq(&self, other: &Hash) -> bool {
        constant_time_eq::constant_time_eq(&self.0[..], &other.0[..])
    }
}

/// This implementation is constant-time.
impl PartialEq<[u8; OUT_LEN]> for Hash {
    fn eq(&self, other: &[u8; OUT_LEN]) -> bool {
        constant_time_eq::constant_time_eq(&self.0[..], other)
    }
}

impl Eq for Hash {}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Hash(0x{})", self.to_hex())
    }
}

/// An extensible-output hash result, from which you can read any number of
/// bytes.
///
/// This is a simple proof of concept. A more fleshed-out implementation would
/// implement `std::io::{Read, Seek}` and would use SIMD parallelism
/// internally.
#[derive(Clone)]
pub struct Output {
    cv: [Word; 8],
    block: [u8; BLOCK_LEN],
    block_len: u8,
    offset: u64,
    flags: Flags,
    platform: Platform,
}

impl Output {
    pub fn read(&mut self) -> [u8; 2 * OUT_LEN] {
        let out = self.platform.compress(
            &self.cv,
            &self.block,
            self.block_len,
            self.offset,
            self.flags.bits(),
        );
        self.offset += 2 * OUT_LEN as u64;
        let mut bytes = [0; 2 * OUT_LEN];
        {
            let refs = mut_array_refs!(&mut bytes, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4);
            *refs.0 = out[0].to_le_bytes();
            *refs.1 = out[1].to_le_bytes();
            *refs.2 = out[2].to_le_bytes();
            *refs.3 = out[3].to_le_bytes();
            *refs.4 = out[4].to_le_bytes();
            *refs.5 = out[5].to_le_bytes();
            *refs.6 = out[6].to_le_bytes();
            *refs.7 = out[7].to_le_bytes();
            *refs.8 = out[8].to_le_bytes();
            *refs.9 = out[9].to_le_bytes();
            *refs.10 = out[10].to_le_bytes();
            *refs.11 = out[11].to_le_bytes();
            *refs.12 = out[12].to_le_bytes();
            *refs.13 = out[13].to_le_bytes();
            *refs.14 = out[14].to_le_bytes();
            *refs.15 = out[15].to_le_bytes();
        }
        bytes
    }

    pub fn to_hash(mut self) -> Hash {
        let out = self.read();
        (*array_ref!(out, 0, OUT_LEN)).into()
    }
}

// Derive an empty Debug impl, because the contents might be secret.
impl fmt::Debug for Output {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Output {{ ... }}")
    }
}

// Benchmarks only.
#[doc(hidden)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub unsafe fn compress_sse41(
    state: &[Word; 8],
    block: &[u8; BLOCK_LEN],
    block_len: u8,
    offset: u64,
    flags: u8,
) -> [Word; 16] {
    sse41::compress(state, block, block_len, offset, flags)
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
    is_root: IsRoot,
    flags: Flags,
    platform: Platform,
) -> Output {
    if let IsRoot::Root = is_root {
        debug_assert_eq!(offset, 0, "root chunk must be offset 0");
    }
    let mut cv = iv(key);
    let mut block_flags = flags | Flags::CHUNK_START;
    let mut block_offset = 0;
    while block_offset + BLOCK_LEN <= chunk.len() {
        if block_offset + BLOCK_LEN == chunk.len() {
            return Output {
                cv,
                block: *array_ref!(chunk, block_offset, BLOCK_LEN),
                block_len: BLOCK_LEN as u8,
                offset,
                flags: block_flags | Flags::CHUNK_END | is_root.flag(),
                platform,
            };
        }
        platform.compress_in_place(
            &mut cv,
            array_ref!(chunk, block_offset, BLOCK_LEN),
            BLOCK_LEN as u8,
            offset,
            block_flags.bits(),
        );
        block_offset += BLOCK_LEN;
        block_flags = flags;
    }

    // There's a partial block left over (or the whole chunk was empty, which
    // only happens when the whole message is empty). Pad the last block with
    // zero bytes return it as an Output.
    let block_len = chunk.len() - block_offset;
    let mut last_block = [0; BLOCK_LEN];
    last_block[..block_len].copy_from_slice(&chunk[block_offset..]);
    Output {
        cv,
        block: last_block,
        block_len: block_len as u8,
        offset,
        // Note that block_flags may still contain CHUNK_START.
        flags: block_flags | Flags::CHUNK_END | is_root.flag(),
        platform,
    }
}

// Use SIMD parallelism to hash multiple chunks at the same time on a single
// thread. Returns the number of chunks hashed. These chunks are never the root
// and never empty.
fn hash_chunks_parallel(
    input: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
    platform: Platform,
    out: &mut [u8],
) -> usize {
    debug_assert!(!input.is_empty(), "empty chunks below the root");
    debug_assert!(input.len() <= MAX_SIMD_DEGREE * CHUNK_LEN);
    debug_assert_eq!(offset % CHUNK_LEN as u64, 0, "invalid offset");

    let mut chunks_exact = input.chunks_exact(CHUNK_LEN);
    let mut chunks_array = ArrayVec::<[&[u8; CHUNK_LEN]; MAX_SIMD_DEGREE]>::new();
    for chunk in &mut chunks_exact {
        chunks_array.push(array_ref!(chunk, 0, CHUNK_LEN));
    }
    platform.hash_many(
        &chunks_array,
        key,
        offset,
        CHUNK_LEN as u64,
        flags.bits(),
        Flags::CHUNK_START.bits(),
        Flags::CHUNK_END.bits(),
        out,
    );

    // Handle the remaining partial chunk, if there is one. Note that the empty
    // chunk (meaning the empty message) is a different codepath.
    let chunks_so_far = chunks_array.len();
    if !chunks_exact.remainder().is_empty() {
        let output = hash_one_chunk(
            chunks_exact.remainder(),
            key,
            offset + chunks_so_far as u64 * CHUNK_LEN as u64,
            IsRoot::NotRoot,
            flags,
            platform,
        );
        *array_mut_ref!(out, chunks_so_far * OUT_LEN, OUT_LEN) = output.to_hash().into();
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
    debug_assert_eq!(child_hashes.len() % OUT_LEN, 0, "wacky hash bytes");
    let num_children = child_hashes.len() / OUT_LEN;
    debug_assert!(num_children >= 2, "not enough children");
    debug_assert!(num_children <= 2 * MAX_SIMD_DEGREE_OR_2, "too many");

    let mut parents_exact = child_hashes.chunks_exact(BLOCK_LEN);
    let mut parents_array = ArrayVec::<[&[u8; BLOCK_LEN]; MAX_SIMD_DEGREE_OR_2]>::new();
    for parent in &mut parents_exact {
        parents_array.push(array_ref!(parent, 0, BLOCK_LEN));
    }
    platform.hash_many(
        &parents_array,
        key,
        0, // Parents have no offset.
        0, // Parents have no offset delta.
        (flags | Flags::PARENT).bits(),
        0, // Parents have no start flags.
        0, // Parents have no end flags.
        out,
    );

    // If there's an odd child left over, it becomes an output.
    let parents_so_far = parents_array.len();
    if !parents_exact.remainder().is_empty() {
        out[parents_so_far * OUT_LEN..][..OUT_LEN].copy_from_slice(parents_exact.remainder());
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
    debug_assert!(content_len > CHUNK_LEN);
    // Subtract 1 to reserve at least one byte for the right side.
    let full_chunks = (content_len - 1) / CHUNK_LEN;
    largest_power_of_two_leq(full_chunks) * CHUNK_LEN
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

// This never does root finalization.
fn hash_recurse(
    input: &[u8],
    key: &[Word; 8],
    offset: u64,
    flags: Flags,
    platform: Platform,
    out: &mut [u8],
) -> usize {
    if input.len() <= platform.simd_degree() * CHUNK_LEN {
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

    // Arrange space for the child outputs. Note that parent hashing uses a
    // minimum simd_degree of 2, to make sure that root finalization doesn't
    // need to happen. See the short-circuit case below that makes this work.
    // (We don't define MAX_SIMD_DEGREE to be 2 in general, because that would
    // restrict multithreading for medium-length inputs.)
    let mut children_array = [0; 2 * MAX_SIMD_DEGREE_OR_2 * OUT_LEN];
    let degree = if left.len() == CHUNK_LEN {
        1 // the "simd_degree=1 and we're at the leaf nodes" case
    } else {
        cmp::max(platform.simd_degree(), 2)
    };
    let (left_out, right_out) = children_array.split_at_mut(degree * OUT_LEN);

    // Recurse! This uses multiple threads if Rayon is enabled.
    let (left_n, right_n) = join(
        || hash_recurse(left, key, offset, flags, platform, left_out),
        || hash_recurse(right, key, right_off, flags, platform, right_out),
    );

    // If simd_degree=1, then we'll have left_n=1 and right_n=1. We don't want
    // to combine them into a single parent node, because that might be the
    // root, and this function doesn't do root finalization. In that case,
    // return the children directly, skipping one level of parent hashing.
    // Callers above will behave as though simd_degree=2, and the root caller
    // will have two children to finalize.
    debug_assert_eq!(left_n, degree);
    debug_assert!(right_n >= 1 && right_n <= left_n);
    if left_n == 1 {
        out[..2 * OUT_LEN].copy_from_slice(&children_array[..2 * OUT_LEN]);
        return 2;
    }

    let num_children = left_n + right_n;
    hash_parents_parallel(
        &children_array[..num_children * OUT_LEN],
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
    debug_assert_eq!(children.len() % OUT_LEN, 0);
    debug_assert!(children.len() >= BLOCK_LEN);
    let mut out_array = [0; MAX_SIMD_DEGREE * OUT_LEN / 2];
    while children.len() > BLOCK_LEN {
        let out_n = hash_parents_parallel(children, key, flags, platform, &mut out_array);
        children[..out_n * OUT_LEN].copy_from_slice(&out_array[..out_n * OUT_LEN]);
        children = &mut children[..out_n * OUT_LEN];
    }
    Output {
        cv: iv(key),
        block: *array_ref!(children, 0, BLOCK_LEN),
        block_len: BLOCK_LEN as u8,
        offset: 0,
        flags: flags | Flags::PARENT | Flags::ROOT,
        platform,
    }
}

fn hash_internal(input: &[u8], key_bytes: &[u8; KEY_LEN], flags: Flags) -> Output {
    let platform = Platform::detect();
    let key_words = words_from_key_bytes(key_bytes);
    if input.len() <= CHUNK_LEN {
        return hash_one_chunk(input, &key_words, 0, IsRoot::Root, flags, platform);
    }
    // See comments in hash_recurse about the _OR_2 here.
    let mut children_array = [0; MAX_SIMD_DEGREE_OR_2 * OUT_LEN];
    let num_children = hash_recurse(input, &key_words, 0, flags, platform, &mut children_array);
    condense_root(
        &mut children_array[..num_children * OUT_LEN],
        &key_words,
        flags,
        platform,
    )
}

/// The default hash function.
pub fn hash(input: &[u8]) -> Hash {
    hash_xof(input).to_hash()
}

/// The default hash function, returning an extensible output.
pub fn hash_xof(input: &[u8]) -> Output {
    hash_internal(input, ZERO_KEY, Flags::empty())
}

/// The hash function with a key.
///
/// This is domain separated from `hash`.
pub fn keyed_hash(input: &[u8], key: &[u8; KEY_LEN]) -> Hash {
    keyed_hash_xof(input, key).to_hash()
}

/// The hash function with a key, returning an extensible output.
///
/// This is domain separated from `hash`.
pub fn keyed_hash_xof(input: &[u8], key: &[u8; KEY_LEN]) -> Output {
    hash_internal(input, key, Flags::KEYED_HASH)
}

/// The key derivation function.
///
/// This is domain separated from `hash` and `keyed_hash`. It's functionally
/// the same as `keyed_hash_xof`, except that `context` is intended to be a
/// hardcoded, application-specific string.
pub fn derive_key(key: &[u8; KEY_LEN], context: &[u8]) -> Output {
    hash_internal(context, key, Flags::DERIVE_KEY)
}

// =======================================================================
// ================== Iterative tree hash implementation =================
// =======================================================================

#[derive(Clone)]
struct ChunkState {
    state: [Word; 8],
    offset: u64,
    count: u16,
    buf: [u8; BLOCK_LEN],
    buf_len: u8,
}

impl ChunkState {
    fn new(key: &[Word; 8], offset: u64) -> Self {
        debug_assert_eq!(offset % CHUNK_LEN as u64, 0);
        Self {
            state: iv(key),
            offset,
            count: 0,
            buf: [0; BLOCK_LEN],
            buf_len: 0,
        }
    }

    // The length of the current chunk, including buffered bytes.
    fn len(&self) -> usize {
        self.count as usize + self.buf_len as usize
    }

    fn fill_buf(&mut self, input: &mut &[u8]) {
        let want = BLOCK_LEN - self.buf_len as usize;
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

    fn update(&mut self, mut input: &[u8], flags: Flags, platform: Platform) {
        if self.buf_len > 0 {
            self.fill_buf(&mut input);
            if !input.is_empty() {
                debug_assert_eq!(self.buf_len as usize, BLOCK_LEN);
                let block_flags = flags | self.maybe_chunk_start_flag(); // borrowck
                platform.compress_in_place(
                    &mut self.state,
                    &self.buf,
                    BLOCK_LEN as u8,
                    self.offset,
                    block_flags.bits(),
                );
                self.count += BLOCK_LEN as u16;
                self.buf_len = 0;
                self.buf = [0; BLOCK_LEN];
            }
        }

        while input.len() > BLOCK_LEN {
            debug_assert_eq!(self.buf_len, 0);
            let block_flags = flags | self.maybe_chunk_start_flag(); // borrowck
            platform.compress_in_place(
                &mut self.state,
                array_ref!(input, 0, BLOCK_LEN),
                BLOCK_LEN as u8,
                self.offset,
                block_flags.bits(),
            );
            self.count += BLOCK_LEN as u16;
            input = &input[BLOCK_LEN..];
        }

        self.fill_buf(&mut input);
        debug_assert!(input.is_empty());
        debug_assert!(self.len() <= CHUNK_LEN);
    }

    // Note that similar to hash_one_chunk, when we use this function
    // internally, we might build Output objects without the root flag and with
    // non-zero offsets. But Output objects returned publicly are always root
    // finalizations starting at offset zero.
    fn finalize(&self, is_root: IsRoot, flags: Flags, platform: Platform) -> Output {
        if let IsRoot::Root = is_root {
            // Root finalization starts with offset 0 to support the XOF, so
            // it's convenient that only the chunk at offset 0 can be the root.
            debug_assert_eq!(self.offset, 0);
        }
        let block_flags = flags | self.maybe_chunk_start_flag() | Flags::CHUNK_END | is_root.flag();
        Output {
            cv: self.state,
            block: self.buf,
            block_len: self.buf_len,
            offset: self.offset,
            flags: block_flags,
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
    left_child: &[u8; OUT_LEN],
    right_child: &[u8; OUT_LEN],
    key: &[Word; 8],
    is_root: IsRoot,
    flags: Flags,
    platform: Platform,
) -> Output {
    let mut block = [0; BLOCK_LEN];
    let refs = mut_array_refs!(&mut block, OUT_LEN, OUT_LEN);
    *refs.0 = *left_child;
    *refs.1 = *right_child;
    Output {
        cv: iv(key),
        block,
        block_len: BLOCK_LEN as u8,
        offset: 0,
        flags: flags | Flags::PARENT | is_root.flag(),
        platform,
    }
}

/// A streaming hash implementation, which can accept any number of writes.
///
/// **Performance note:** Using
/// [`std::io::copy`](https://doc.rust-lang.org/std/io/fn.copy.html) together
/// with `Hasher` will generally give poor performance, because it uses a copy
/// buffer that's too small to drive more than a couple SIMD lanes. Use the
/// [`copy_wide`](copy/fn.copy_wide.html) utility function instead.
#[derive(Clone)]
pub struct Hasher {
    subtree_hashes: ArrayVec<[[u8; OUT_LEN]; MAX_DEPTH]>,
    chunk: ChunkState,
    key: [Word; 8],
    flags: Flags,
    platform: Platform,
}

impl Hasher {
    fn new_internal(key_bytes: &[u8; KEY_LEN], flags: Flags) -> Self {
        let key_words = words_from_key_bytes(key_bytes);
        Self {
            subtree_hashes: ArrayVec::new(),
            chunk: ChunkState::new(&key_words, 0),
            key: key_words,
            flags,
            platform: Platform::detect(),
        }
    }

    /// Construct an incremental hasher for the default hash function.
    pub fn new() -> Self {
        Self::new_internal(ZERO_KEY, Flags::empty())
    }

    /// Construct an incremental hasher with a key.
    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        Self::new_internal(key, Flags::KEYED_HASH)
    }

    /// Construct an incremental hasher for key derivation.
    ///
    /// Note that the input in this case is intended to be an
    /// application-specific context string. Most callers should hardcode such
    /// strings and prefer the `derive_key` function.
    pub fn new_derive_key(key: &[u8; KEY_LEN]) -> Self {
        Self::new_internal(key, Flags::DERIVE_KEY)
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
        let total_chunks = total_bytes / CHUNK_LEN as u64;
        self.subtree_hashes.len() > total_chunks.count_ones() as usize
    }

    // Take two subtree hashes off the end of the stack, hash them into a
    // parent node, and put that hash back on the stack.
    fn merge_parent(&mut self) {
        let num_subtrees = self.subtree_hashes.len();
        debug_assert!(num_subtrees >= 2);
        let parent_hash = hash_one_parent(
            &self.subtree_hashes[num_subtrees - 2],
            &self.subtree_hashes[num_subtrees - 1],
            &self.key,
            IsRoot::NotRoot,
            self.flags,
            self.platform,
        )
        .to_hash();
        self.subtree_hashes[num_subtrees - 2] = parent_hash.into();
        self.subtree_hashes.truncate(num_subtrees - 1)
    }

    fn push_chunk_hash(&mut self, hash: &[u8; OUT_LEN], offset: u64) {
        // Do subtree merging *before* pushing the new hash. That way we know
        // that the merged parent node isn't the root.
        while self.needs_merge(offset) {
            self.merge_parent();
        }
        self.subtree_hashes.push(*hash);
    }

    /// Add input bytes to the hash.
    ///
    /// Updating `Hasher` is more efficient when you use a buffer size that's a
    /// multiple of [`WIDE_BUF_LEN`](copy/constant.WIDE_BUF_LEN.html). The
    /// [`copy_wide`](copy/fn.copy_wide.html) helper function takes care of
    /// this.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        // When we have whole chunks coming in, hash_many gives the best
        // performance. However, we have to be careful with the very first
        // chunk. If we don't have more input yet, then we don't know whether
        // it's the root, and we have to keep it in the ChunkState until we
        // find out. Also, if we have any partial chunk bytes in the ChunkState
        // already, we need to finish it.
        let is_first_chunk = self.chunk.offset == 0;
        let maybe_root = is_first_chunk && input.len() == CHUNK_LEN;
        if maybe_root || self.chunk.len() > 0 {
            let want = CHUNK_LEN - self.chunk.len();
            let take = cmp::min(want, input.len());
            self.chunk.update(&input[..take], self.flags, self.platform);
            input = &input[take..];
            if !input.is_empty() {
                // We've filled the current chunk, and there's more input
                // coming, so we know it's not the root and we can finalize it.
                // Then we'll proceed to hashing whole chunks below.
                debug_assert_eq!(self.chunk.len(), CHUNK_LEN);
                let chunk_hash = self
                    .chunk
                    .finalize(IsRoot::NotRoot, self.flags, self.platform)
                    .to_hash();
                self.push_chunk_hash(chunk_hash.as_bytes(), self.chunk.offset);
                self.chunk = ChunkState::new(&self.key, self.chunk.offset + CHUNK_LEN as u64);
            } else {
                return self;
            }
        }

        // At this point, the ChunkState is clear. Hash all the full chunks
        // that we can, using SIMD parallelism. But note that this still
        // single-threaded. (Only the recursive hash functions are
        // multi-threaded.)
        let mut chunks_exact = input.chunks_exact(CHUNK_LEN);
        let mut fused_chunks = chunks_exact
            .by_ref()
            .map(|chunk| array_ref!(chunk, 0, CHUNK_LEN))
            .fuse();
        let mut chunks_array = ArrayVec::<[&[u8; CHUNK_LEN]; MAX_SIMD_DEGREE]>::new();
        let mut out_array = [0; MAX_SIMD_DEGREE * OUT_LEN];
        debug_assert_eq!(self.chunk.len(), 0);
        loop {
            chunks_array.extend(fused_chunks.by_ref().take(MAX_SIMD_DEGREE));
            if chunks_array.is_empty() {
                // We've exhausted the whole chunks. Add any remaining input to
                // the ChunkState below.
                break;
            }
            self.platform.hash_many(
                &chunks_array,
                &self.key,
                self.chunk.offset,
                CHUNK_LEN as u64,
                self.flags.bits(),
                Flags::CHUNK_START.bits(),
                Flags::CHUNK_END.bits(),
                &mut out_array,
            );
            for chunk_hash in out_array.chunks_exact(OUT_LEN).take(chunks_array.len()) {
                self.push_chunk_hash(array_ref!(chunk_hash, 0, OUT_LEN), self.chunk.offset);
                // Move the ChunkState's offset forward after each hash. This
                // is safe because it's clear, and it leaves it in the right
                // position for any remainder bytes below or subsequent calls
                // to update.
                self.chunk.offset += CHUNK_LEN as u64;
            }
            chunks_array.clear();
        }

        // The loop above hashed all the whole chunks it could. If there are
        // any bytes left, add them to the ChunkState. The ChunkState's offset
        // may have been updated above.
        debug_assert_eq!(self.chunk.len(), 0);
        debug_assert!(chunks_exact.remainder().len() < CHUNK_LEN);
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
                .update(chunks_exact.remainder(), self.flags, self.platform);
        }
        self
    }

    /// Finalize the root hash.
    ///
    /// This method is idempotent, and calling it multiple times will give the
    /// same result. It's also possible to add more input and finalize again.
    pub fn finalize(&self) -> Hash {
        self.finalize_xof().to_hash()
    }

    /// Finalize the root hash, returning an extensible output.
    ///
    /// This method is idempotent, and calling it multiple times will give the
    /// same result. It's also possible to add more input and finalize again.
    pub fn finalize_xof(&self) -> Output {
        // If the current chunk is the only chunk, that makes it the root node
        // also. Convert it directly into an Output. Otherwise, we need to
        // merge subtrees below.
        if self.subtree_hashes.is_empty() {
            return self.chunk.finalize(IsRoot::Root, self.flags, self.platform);
        }

        // If there are any bytes in the ChunkState, finalize that chunk and
        // merge it with everything in the subtree stack. In that case, the
        // work we did at the end of update above guarantees that the stack
        // doesn't contain any unmerged subtrees that need to be merged first.
        // (This is important, because if there were two chunk hashes sitting
        // on top of the stack, they would need to merge with each other, and
        // merging a new chunk hash into them would be incorrect.)
        //
        // If there are no bytes in the ChunkState, we'll merge what's already
        // in the stack. In this case it's fine if there are unmerged chunks on
        // top, because we'll merge them with each other.
        let mut working_hash: [u8; OUT_LEN];
        let mut next_subtree_index: usize;
        if self.chunk.len() > 0 {
            debug_assert!(!self.needs_merge(self.chunk.offset));
            working_hash = self
                .chunk
                .finalize(IsRoot::NotRoot, self.flags, self.platform)
                .to_hash()
                .into();
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
            let output = hash_one_parent(
                &self.subtree_hashes[next_subtree_index],
                &working_hash,
                &self.key,
                is_root,
                self.flags,
                self.platform,
            );
            if next_subtree_index == 0 {
                return output;
            }
            working_hash = output.to_hash().into();
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

impl std::io::Write for Hasher {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        self.update(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
