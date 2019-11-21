#! /usr/bin/env python3

IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]

MSG_SCHEDULE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
]

BLOCK_LEN = 64
CHUNK_LEN = 2048
KEY_LEN = 32
OUT_LEN = 32
WORD_BITS = 32
WORD_BYTES = 4
WORD_MAX = 2**WORD_BITS - 1

# domain flags
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
PARENT = 1 << 2
ROOT = 1 << 3
KEYED_HASH = 1 << 4
DERIVE_KEY = 1 << 5


def rotate_right(x, n):
    return (x >> n | x << (WORD_BITS - n)) & WORD_MAX


def g(state, a, b, c, d, x, y):
    state[a] = (state[a] + state[b] + x) & WORD_MAX
    state[d] = rotate_right(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & WORD_MAX
    state[b] = rotate_right(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + y) & WORD_MAX
    state[d] = rotate_right(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & WORD_MAX
    state[b] = rotate_right(state[b] ^ state[c], 7)


def round(state, msg_words, schedule):
    # Mix the columns.
    g(state, 0, 4, 8, 12, msg_words[schedule[0]], msg_words[schedule[1]])
    g(state, 1, 5, 9, 13, msg_words[schedule[2]], msg_words[schedule[3]])
    g(state, 2, 6, 10, 14, msg_words[schedule[4]], msg_words[schedule[5]])
    g(state, 3, 7, 11, 15, msg_words[schedule[6]], msg_words[schedule[7]])
    # Mix the rows.
    g(state, 0, 5, 10, 15, msg_words[schedule[8]], msg_words[schedule[9]])
    g(state, 1, 6, 11, 12, msg_words[schedule[10]], msg_words[schedule[11]])
    g(state, 2, 7, 8, 13, msg_words[schedule[12]], msg_words[schedule[13]])
    g(state, 3, 4, 9, 14, msg_words[schedule[14]], msg_words[schedule[15]])


def words_from_bytes(buf):
    words = [0] * (len(buf) // WORD_BYTES)
    for word_i in range(len(words)):
        words[word_i] = int.from_bytes(
            buf[word_i * WORD_BYTES:(word_i + 1) * WORD_BYTES], "little")
    return words


def bytes_from_words(words):
    buf = bytearray(OUT_LEN)
    for word_i in range(len(words)):
        buf[WORD_BYTES * word_i:WORD_BYTES * (word_i + 1)] = \
            words[word_i].to_bytes(WORD_BYTES, "little")
    return buf


def offset_low(offset):
    return offset & WORD_MAX


def offset_high(offset):
    return (offset >> WORD_BITS) & WORD_MAX


def compress_inner(cv, block, block_len, offset, flags):
    block_words = words_from_bytes(block)
    state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        offset_low(offset),
        offset_high(offset),
        block_len,
        flags,
    ]
    for round_number in range(7):
        round(state, block_words, MSG_SCHEDULE[round_number])
    return state


# The standard compression function. Used for chaining values in the interior
# of the tree, and to compute the default OUT_LEN output. This updates the
# chaining value in place. Note that the output of compress() is a prefix of
# the output of compress_xof().
def compress(cv, block, block_len, offset, flags):
    state = compress_inner(cv, block, block_len, offset, flags)
    cv[0] = state[0] ^ state[8]
    cv[1] = state[1] ^ state[9]
    cv[2] = state[2] ^ state[10]
    cv[3] = state[3] ^ state[11]
    cv[4] = state[4] ^ state[12]
    cv[5] = state[5] ^ state[13]
    cv[6] = state[6] ^ state[14]
    cv[7] = state[7] ^ state[15]


# The wide compression function. Used to compute XOF output larger than
# OUT_LEN. This returns output bytes without modifying the chaining value that
# produced it. Note that the output of compress() is a prefix of the output of
# compress_xof().
def compress_xof(cv, block, block_len, offset, flags):
    state = compress_inner(cv, block, block_len, offset, flags)
    state[0] ^= state[8]
    state[1] ^= state[9]
    state[2] ^= state[10]
    state[3] ^= state[11]
    state[4] ^= state[12]
    state[5] ^= state[13]
    state[6] ^= state[14]
    state[7] ^= state[15]
    state[8] ^= cv[0]
    state[9] ^= cv[1]
    state[10] ^= cv[2]
    state[11] ^= cv[3]
    state[12] ^= cv[4]
    state[13] ^= cv[5]
    state[14] ^= cv[6]
    state[15] ^= cv[7]
    return bytes_from_words(state)


# The XOF output object. This can provide any number of output bytes. Note that
# Output objects returned from public functions always set the ROOT flag, but
# Output objects used internally do not.
class Output:
    def __init__(self, cv, block, block_len, offset, flags):
        self.cv = cv
        self.block = block
        self.block_len = block_len
        self.offset = offset
        self.flags = flags

    # Return a hash of the default size, OUT_LEN.
    def to_hash(self):
        words = self.cv[:]
        compress(words, self.block, self.block_len, self.offset, self.flags)
        return bytes_from_words(words)

    # Return any number of output bytes. Note that this interface doesn't
    # mutate the Output object, so calling it twice will give the same bytes.
    def to_bytes(self, num_bytes):
        buf = bytearray(num_bytes)
        offset = self.offset
        i = 0
        while i < num_bytes:
            output = compress_xof(self.cv, self.block, self.block_len, offset,
                                  self.flags)
            take = min(len(output), num_bytes - i)
            buf[i:i + take] = output[:take]
            offset += len(output)
            i += len(output)
        return buf


# Hash a node, which might be either a parent or a chunk, and return an Output
# object. Parent nodes will always be exactly one block, but chunks may be more
# than one. The `flags` argument applies to all blocks in a chunk, while
# `flags_start` is only included for the first block, and `flags_end` is only
# included for the last block. Note that if a chunk is just a single block,
# that block will get both start and end flags.
def hash_node(node, key_words, offset, flags, flags_start, flags_end):
    cv = key_words[:]
    block_flags = flags | flags_start
    position = 0
    while len(node) - position > BLOCK_LEN:
        block = node[position:position + BLOCK_LEN]
        compress(cv, block, BLOCK_LEN, offset, block_flags)
        block_flags = flags
        position += BLOCK_LEN
    block_len = len(node) - position
    block = bytearray(BLOCK_LEN)
    block[0:block_len] = node[position:]
    block_flags |= flags_end
    return Output(cv, block, block_len, offset, block_flags)


# The left subtree is the largest possible complete tree that still leaves at
# least one byte for the right subtree. That is, the number of chunks in the
# left subtree is the largest power of two that fits.
def left_len(parent_len):
    available_chunks = (parent_len - 1) // CHUNK_LEN
    power_of_two_chunks = 2**(available_chunks.bit_length() - 1)
    return CHUNK_LEN * power_of_two_chunks


# Hash an entire subtree recursively, returning an Output object.
def hash_recurse(input_bytes, key_words, offset, flags, is_root):
    maybe_root = ROOT if is_root else 0
    # If this subtree is just one chunk, hash it as a single node and return.
    # Note that if that chunk is the root, the root flag will only be set for
    # the final block.
    if len(input_bytes) <= CHUNK_LEN:
        return hash_node(input_bytes, key_words, offset, flags, CHUNK_START,
                         CHUNK_END | maybe_root)
    # The subtree is larger than one chunk. Split it into left and right
    # subtrees, recurse to compute their hashes, and combine those hashes into
    # a parent node.
    left = input_bytes[:left_len(len(input_bytes))]
    right = input_bytes[len(left):]
    right_offset = offset + len(left)
    # Note that the left and right subtrees always use is_root=False.
    left_hash = hash_recurse(left, key_words, offset, flags, False).to_hash()
    right_hash = hash_recurse(right, key_words, right_offset, flags,
                              False).to_hash()
    node_bytes = left_hash + right_hash
    # Parent nodes always use an offset of 0. And because they're a single
    # block, they don't need start or end flags, so those are 0 too.
    parent_flags = flags | PARENT | maybe_root
    return hash_node(node_bytes, key_words, 0, parent_flags, 0, 0)


# The core hash function, taking an input of any length, a 32-byte key, and any
# domain separation flags that will apply to all nodes (namely KEYED_HASH or
# DERIVE_KEY). Returns an Output object.
def hash_internal(input_bytes, key_words, flags):
    return hash_recurse(input_bytes, key_words, 0, flags, True)


# ==================== Public API ====================


# The default hash function, returning a 32-byte hash.
def hash(input_bytes):
    return hash_xof(input_bytes).to_hash()


# The default hash function, returning an extensible Output object.
def hash_xof(input_bytes):
    return hash_internal(input_bytes, IV, 0)


# The keyed hash function, returning a 32-byte hash.
def keyed_hash(key_bytes, input_bytes):
    return keyed_hash_xof(key_bytes, input_bytes).to_hash()


# The keyed hash function, returning an extensible Output object.
def keyed_hash_xof(key_bytes, input_bytes):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(input_bytes, key_words, KEYED_HASH)


# The KDF, returning a 32 byte key.
def derive_key(key_bytes, context_bytes):
    return derive_key_xof(key_bytes, context_bytes).to_hash()


# The KDF, returning an extensible Output object.
def derive_key_xof(key_bytes, context_bytes):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(context_bytes, key_words, DERIVE_KEY)
