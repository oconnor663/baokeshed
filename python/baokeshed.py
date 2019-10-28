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
CHUNK_LEN = 4096
KEY_LEN = 32
OUT_LEN = 32
WORD_BITS = 32
WORD_BYTES = 4
WORD_MAX = 2**WORD_BITS - 1

# domain flags
ROOT = 1 << 0
PARENT = 1 << 1
CHUNK_END = 1 << 2
CHUNK_START = 1 << 3
KEYED_HASH = 1 << 4
DERIVE_KEY = 1 << 5


def rotate_right(x, n):
    return (x >> n | x << (32 - n)) & WORD_MAX


def g(state, a, b, c, d, x, y):
    state[a] = (state[a] + state[b] + x) & WORD_MAX
    state[d] = rotate_right(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & WORD_MAX
    state[b] = rotate_right(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + y) & WORD_MAX
    state[d] = rotate_right(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & WORD_MAX
    state[b] = rotate_right(state[b] ^ state[c], 7)


def round(state, msg_words, r):
    schedule = MSG_SCHEDULE[r]
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
        buf[WORD_BYTES * word_i:WORD_BYTES *
            (word_i + 1)] = words[word_i].to_bytes(WORD_BYTES, "little")
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
        IV[4] ^ offset_low(offset),
        IV[5] ^ offset_high(offset),
        IV[6] ^ block_len,
        IV[7] ^ flags,
    ]
    for r in range(7):
        round(state, block_words, r)
    return state


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
    compress(cv, block, block_len, offset, block_flags | flags_end)
    return bytes_from_words(cv)


# Left subtrees contain the largest possible power of two number of chunks,
# with at least one byte left for the right subtree.
def left_len(parent_len):
    available_chunks = (parent_len - 1) // CHUNK_LEN
    power_of_two_chunks = 2**(available_chunks.bit_length() - 1)
    return CHUNK_LEN * power_of_two_chunks


def hash_recurse(input_bytes, key_words, offset, flags, is_root):
    maybe_root = ROOT if is_root else 0
    if len(input_bytes) <= CHUNK_LEN:
        return hash_node(input_bytes, key_words, offset, flags, CHUNK_START,
                         CHUNK_END | maybe_root)
    left = input_bytes[:left_len(len(input_bytes))]
    right = input_bytes[len(left):]
    right_offset = offset + len(left)
    left_hash = hash_recurse(left, key_words, offset, flags, False)
    right_hash = hash_recurse(right, key_words, right_offset, flags, False)
    node_bytes = left_hash + right_hash
    return hash_node(node_bytes, key_words, 0, flags | PARENT | maybe_root, 0,
                     0)


def hash_internal(input_bytes, key_words, flags):
    return hash_recurse(input_bytes, key_words, 0, flags, True)


def hash(input_bytes):
    return hash_internal(input_bytes, IV, 0)


def keyed_hash(input_bytes, key_bytes):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(input_bytes, key_words, KEYED_HASH)


def derive_key(key_bytes, context_bytes):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(context_bytes, key_words, DERIVE_KEY)
