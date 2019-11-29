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


def wrapping_add(a, b):
    return (a + b) & WORD_MAX


def rotate_right(x, n):
    return (x >> n | x << (WORD_BITS - n)) & WORD_MAX


def g(state, a, b, c, d, x, y):
    state[a] = wrapping_add(state[a], wrapping_add(state[b], x))
    state[d] = rotate_right(state[d] ^ state[a], 16)
    state[c] = wrapping_add(state[c], state[d])
    state[b] = rotate_right(state[b] ^ state[c], 12)
    state[a] = wrapping_add(state[a], wrapping_add(state[b], y))
    state[d] = rotate_right(state[d] ^ state[a], 8)
    state[c] = wrapping_add(state[c], state[d])
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
    buf = bytearray(len(words) * WORD_BYTES)
    for word_i in range(len(words)):
        buf[WORD_BYTES * word_i:WORD_BYTES * (word_i + 1)] = \
            words[word_i].to_bytes(WORD_BYTES, "little")
    return buf


def compress(cv, block, block_len, offset, flags):
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
        offset & WORD_MAX,
        (offset >> WORD_BITS) & WORD_MAX,
        block_len,
        flags,
    ]
    for round_number in range(7):
        round(state, block_words, MSG_SCHEDULE[round_number])
    for i in range(8):
        state[i] ^= state[i + 8]
        state[i + 8] ^= cv[i]
    return state


# Each chunk or parent node can produce either an 32-byte chaining value or, by
# setting the ROOT flag, any number of final output bytes. The Output struct
# captures the state just prior to choosing between those two possibilities.
class Output:
    def __init__(self, cv, block, block_len, offset, flags):
        self.cv = cv
        self.block = block
        self.block_len = block_len
        self.offset = offset
        self.flags = flags

    def chaining_value(self):
        words = compress(self.cv, self.block, self.block_len, self.offset,
                         self.flags)
        return bytes_from_words(words[:8])

    def root_output_bytes(self, out_len):
        buf = bytearray(out_len)
        offset = 0
        while offset < out_len:
            words = compress(self.cv, self.block, self.block_len, offset,
                             self.flags | ROOT)
            out_bytes = bytes_from_words(words)
            take = min(len(out_bytes), out_len - offset)
            buf[offset:offset + take] = out_bytes[:take]
            offset += take
        return buf


def hash_chunk(chunk, key_words, offset, flags):
    cv = key_words
    block_flags = flags | CHUNK_START
    position = 0
    while len(chunk) - position > BLOCK_LEN:
        block = chunk[position:position + BLOCK_LEN]
        cv = compress(cv, block, BLOCK_LEN, offset, block_flags)[:8]
        block_flags = flags
        position += BLOCK_LEN
    block_len = len(chunk) - position
    block = bytearray(BLOCK_LEN)
    block[0:block_len] = chunk[position:]
    block_flags |= CHUNK_END
    return Output(cv, block, block_len, offset, block_flags)


def hash_parent(left_child, right_child, key_words, flags):
    node_bytes = left_child.chaining_value() + right_child.chaining_value()
    return Output(key_words, node_bytes, BLOCK_LEN, 0, PARENT | flags)


# The left subtree is the largest possible complete tree that still leaves at
# least one byte for the right subtree. That is, the number of chunks in the
# left subtree is the largest power of two that fits.
def left_len(parent_len):
    available_chunks = (parent_len - 1) // CHUNK_LEN
    power_of_two_chunks = 2**(available_chunks.bit_length() - 1)
    return CHUNK_LEN * power_of_two_chunks


# Hash an entire subtree recursively, returning an Output object.
def hash_recurse(input_bytes, key_words, offset, flags):
    if len(input_bytes) <= CHUNK_LEN:
        return hash_chunk(input_bytes, key_words, offset, flags)
    # The subtree is larger than one chunk. Split it into left and right
    # subtrees, recurse to compute their chaining values, and combine those
    # chaining values into a parent output.
    left = input_bytes[:left_len(len(input_bytes))]
    right = input_bytes[len(left):]
    right_offset = offset + len(left)
    left_output = hash_recurse(left, key_words, offset, flags)
    right_output = hash_recurse(right, key_words, right_offset, flags)
    return hash_parent(left_output, right_output, key_words, flags)


def hash_internal(input_bytes, key_words, flags, out_len):
    output = hash_recurse(input_bytes, key_words, 0, flags)
    return output.root_output_bytes(out_len)


# ==================== Public API ====================


def hash(input_bytes, out_len):
    return hash_internal(input_bytes, IV, 0, out_len)


def keyed_hash(key_bytes, input_bytes, out_len):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(input_bytes, key_words, KEYED_HASH, out_len)


def derive_key(key_bytes, context_bytes, out_len):
    key_words = words_from_bytes(key_bytes)
    return hash_internal(context_bytes, key_words, DERIVE_KEY, out_len)
