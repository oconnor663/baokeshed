// NB: This is only for benchmarking. The guy who wrote this file hasn't
// touched C since college. Please don't use this code in production.

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// size constants
#define BLOCK_LEN 64
#define CHUNK_LEN 4096
#define KEY_LEN 32
#define OUT_LEN 32
#define MAX_DEPTH 52

// internal flags
#define ROOT 128
#define PARENT 64
#define CHUNK_END 32
#define CHUNK_START 16

const uint32_t IV[8] = {0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
                        0x510E527FUL, 0x9B05688CUL, 0x1F83D9ABUL, 0x5BE0CD19UL};

const uint8_t MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
};

static inline uint32_t load32(const void *src) {
  const uint8_t *p = (const uint8_t *)src;
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
}

static inline void store32(void *dst, uint32_t w) {
  uint8_t *p = (uint8_t *)dst;
  p[0] = (uint8_t)(w >> 0);
  p[1] = (uint8_t)(w >> 8);
  p[2] = (uint8_t)(w >> 16);
  p[3] = (uint8_t)(w >> 24);
}

static inline uint32_t rotr32(uint32_t w, uint32_t c) {
  return (w >> c) | (w << (32 - c));
}

// Count the number of 1 bits.
static inline uint8_t popcnt(uint64_t x) {
  uint8_t count = 0;
  while (x > 0) {
    count += ((uint8_t)x) & 1;
    x >>= 1;
  }
  return count;
}

static inline void g(uint32_t *state, size_t a, size_t b, size_t c, size_t d,
                     uint32_t x, uint32_t y) {
  state[a] = state[a] + state[b] + x;
  state[d] = rotr32(state[d] ^ state[a], 16);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 12);
  state[a] = state[a] + state[b] + y;
  state[d] = rotr32(state[d] ^ state[a], 8);
  state[c] = state[c] + state[d];
  state[b] = rotr32(state[b] ^ state[c], 7);
}

static inline void round_fn(uint32_t *state, const uint32_t *msg,
                            size_t round) {
  // Select the message schedule based on the round.
  const uint8_t *schedule = MSG_SCHEDULE[round];

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

static inline uint32_t offset_low(uint64_t offset) { return (uint32_t)offset; }

static inline uint32_t offset_high(uint64_t offset) {
  return (uint32_t)(offset >> 32);
}

static inline uint32_t block_flags(uint8_t block_len, uint8_t internal_flags) {
  // The lower bits of the block flags word are the block length. The higher
  // bits are the internal flags (ROOT etc.). The middle bits are currently
  // unused.
  return ((uint32_t)block_len) | (((uint32_t)internal_flags) << 24);
}

static inline void load_msg_words(const uint8_t block[BLOCK_LEN],
                                  uint32_t words[8]) {
  words[0] = load32(block + 4 * 0);
  words[1] = load32(block + 4 * 1);
  words[2] = load32(block + 4 * 2);
  words[3] = load32(block + 4 * 3);
  words[4] = load32(block + 4 * 4);
  words[5] = load32(block + 4 * 5);
  words[6] = load32(block + 4 * 6);
  words[7] = load32(block + 4 * 7);
  words[8] = load32(block + 4 * 8);
  words[9] = load32(block + 4 * 9);
  words[10] = load32(block + 4 * 10);
  words[11] = load32(block + 4 * 11);
  words[12] = load32(block + 4 * 12);
  words[13] = load32(block + 4 * 13);
  words[14] = load32(block + 4 * 14);
  words[15] = load32(block + 4 * 15);
}

static inline void load_key_words(const uint8_t key[KEY_LEN],
                                  uint32_t words[8]) {
  words[0] = load32(key + 4 * 0);
  words[1] = load32(key + 4 * 1);
  words[2] = load32(key + 4 * 2);
  words[3] = load32(key + 4 * 3);
  words[4] = load32(key + 4 * 4);
  words[5] = load32(key + 4 * 5);
  words[6] = load32(key + 4 * 6);
  words[7] = load32(key + 4 * 7);
}

static inline void write_state_bytes(const uint32_t state[8],
                                     uint8_t out[OUT_LEN]) {
  store32(&out[4 * 0], state[0]);
  store32(&out[4 * 1], state[1]);
  store32(&out[4 * 2], state[2]);
  store32(&out[4 * 3], state[3]);
  store32(&out[4 * 4], state[4]);
  store32(&out[4 * 5], state[5]);
  store32(&out[4 * 6], state[6]);
  store32(&out[4 * 7], state[7]);
}

void compress(uint32_t state[8], const uint8_t block[BLOCK_LEN],
              uint8_t block_len, uint64_t offset, uint8_t internal_flags,
              uint32_t context) {
  uint32_t block_words[16];
  load_msg_words(block, block_words);

  uint32_t full_state[16] = {
      state[0],
      state[1],
      state[2],
      state[3],
      state[4],
      state[5],
      state[6],
      state[7],
      IV[0],
      IV[1],
      IV[2],
      IV[3],
      IV[4] ^ offset_low(offset),
      IV[5] ^ offset_high(offset),
      IV[6] ^ block_flags(block_len, internal_flags),
      IV[7] ^ context,
  };

  round_fn(&full_state[0], &block_words[0], 0);
  round_fn(&full_state[0], &block_words[0], 1);
  round_fn(&full_state[0], &block_words[0], 2);
  round_fn(&full_state[0], &block_words[0], 3);
  round_fn(&full_state[0], &block_words[0], 4);
  round_fn(&full_state[0], &block_words[0], 5);
  round_fn(&full_state[0], &block_words[0], 6);

  state[0] = full_state[0] ^ full_state[8];
  state[1] = full_state[1] ^ full_state[9];
  state[2] = full_state[2] ^ full_state[10];
  state[3] = full_state[3] ^ full_state[11];
  state[4] = full_state[4] ^ full_state[12];
  state[5] = full_state[5] ^ full_state[13];
  state[6] = full_state[6] ^ full_state[14];
  state[7] = full_state[7] ^ full_state[15];
}

static inline void init_iv(const uint32_t key_words[8], uint32_t state[8]) {
  state[0] = IV[0] ^ key_words[0];
  state[1] = IV[1] ^ key_words[1];
  state[2] = IV[2] ^ key_words[2];
  state[3] = IV[3] ^ key_words[3];
  state[4] = IV[4] ^ key_words[4];
  state[5] = IV[5] ^ key_words[5];
  state[6] = IV[6] ^ key_words[6];
  state[7] = IV[7] ^ key_words[7];
}

typedef struct {
  uint32_t state[8];
  uint64_t offset;
  uint16_t count;
  uint8_t buf[BLOCK_LEN];
  uint8_t buf_len;
} chunk_state;

void chunk_state_init(chunk_state *self, const uint32_t key_words[8],
                      uint64_t offset) {
  init_iv(key_words, self->state);
  self->offset = offset;
  self->count = 0;
  self->buf_len = 0;
  memset(self->buf, 0, BLOCK_LEN);
}

size_t chunk_state_len(const chunk_state *self) {
  return ((size_t)self->count) + ((size_t)self->buf_len);
}

size_t chunk_state_fill_buf(chunk_state *self, const uint8_t *input,
                            size_t input_len) {
  size_t take = BLOCK_LEN - ((size_t)self->buf_len);
  if (take > input_len) {
    take = input_len;
  }
  uint8_t *dest = self->buf + ((size_t)self->buf_len);
  memcpy(dest, input, take);
  self->buf_len += (uint8_t)take;
  return take;
}

uint8_t chunk_state_maybe_start_flag(const chunk_state *self) {
  if (self->count == 0) {
    return CHUNK_START;
  } else {
    return 0;
  }
}

void chunk_state_update(chunk_state *self, const uint8_t *input,
                        size_t input_len, uint32_t context) {
  if (self->buf_len > 0) {
    size_t take = chunk_state_fill_buf(self, input, input_len);
    input += take;
    input_len -= take;
    if (input_len > 0) {
      compress(self->state, self->buf, BLOCK_LEN, self->offset,
               chunk_state_maybe_start_flag(self), context);
      self->count += (uint16_t)BLOCK_LEN;
      self->buf_len = 0;
      memset(self->buf, 0, BLOCK_LEN);
    }
  }

  while (input_len > BLOCK_LEN) {
    compress(self->state, input, BLOCK_LEN, self->offset,
             chunk_state_maybe_start_flag(self), context);
    self->count += (uint16_t)BLOCK_LEN;
    input += BLOCK_LEN;
    input_len -= BLOCK_LEN;
  }

  size_t take = chunk_state_fill_buf(self, input, input_len);
  input += take;
  input_len -= take;
}

void chunk_state_finalize(const chunk_state *self, uint32_t context,
                          bool is_root, uint8_t out[OUT_LEN]) {
  uint32_t state_copy[8];
  memcpy(state_copy, self->state, sizeof(state_copy));
  uint8_t flags = chunk_state_maybe_start_flag(self) | CHUNK_END;
  if (is_root) {
    flags |= ROOT;
  }
  compress(state_copy, self->buf, self->buf_len, self->offset, flags, context);
  write_state_bytes(state_copy, out);
}

void hash_one_parent(const uint8_t block[BLOCK_LEN],
                     const uint32_t key_words[8], bool is_root,
                     uint32_t context, uint8_t out[OUT_LEN]) {
  uint8_t flags = PARENT;
  if (is_root) {
    flags |= ROOT;
  }
  uint32_t state[8];
  init_iv(key_words, state);
  compress(state, block, BLOCK_LEN, 0, flags, context);
  write_state_bytes(state, out);
}

typedef struct {
  chunk_state chunk;
  uint32_t key_words[8];
  uint32_t context;
  uint8_t subtree_hashes_len;
  uint8_t subtree_hashes[MAX_DEPTH * OUT_LEN];
} hasher;

void hasher_init(hasher *self, const uint8_t key[KEY_LEN], uint32_t context) {
  load_key_words(key, self->key_words);
  chunk_state_init(&self->chunk, self->key_words, 0);
  self->context = context;
  self->subtree_hashes_len = 0;
}

bool hasher_needs_merge(const hasher *self, uint64_t total_bytes) {
  uint64_t total_chunks = total_bytes / CHUNK_LEN;
  return self->subtree_hashes_len > popcnt(total_chunks);
}

void hasher_merge_parent(hasher *self) {
  size_t parent_block_start =
      (((size_t)self->subtree_hashes_len) - 2) * OUT_LEN;
  uint8_t out[OUT_LEN];
  hash_one_parent(&self->subtree_hashes[parent_block_start], self->key_words,
                  false, self->context, out);
  memcpy(&self->subtree_hashes[parent_block_start], out, OUT_LEN);
  self->subtree_hashes_len -= 1;
}

void hasher_push_chunk_hash(hasher *self, uint8_t hash[OUT_LEN],
                            uint64_t offset) {
  assert(self->subtree_hashes_len < MAX_DEPTH);
  while (hasher_needs_merge(self, offset)) {
    hasher_merge_parent(self);
  }
  memcpy(&self->subtree_hashes[self->subtree_hashes_len * OUT_LEN], hash,
         OUT_LEN);
  self->subtree_hashes_len += 1;
}

void hasher_update(hasher *self, const void *input, size_t input_len) {
  uint8_t *input_bytes = (uint8_t *)input;
  while (input_len > 0) {
    if (chunk_state_len(&self->chunk) == CHUNK_LEN) {
      uint8_t hash[OUT_LEN];
      chunk_state_finalize(&self->chunk, self->context, false, hash);
      hasher_push_chunk_hash(self, hash, self->chunk.offset);
      uint64_t new_offset = self->chunk.offset + CHUNK_LEN;
      chunk_state_init(&self->chunk, self->key_words, new_offset);
      // Work ahead, to simplify finalize().
      while (hasher_needs_merge(self, new_offset)) {
        hasher_merge_parent(self);
      }
    }
    size_t take = CHUNK_LEN - chunk_state_len(&self->chunk);
    if (take > input_len) {
      take = input_len;
    }
    chunk_state_update(&self->chunk, input_bytes, take, self->context);
    input_bytes += take;
    input_len -= take;
  }
}

void hasher_finalize(const hasher *self, uint8_t out[OUT_LEN]) {
  if (self->subtree_hashes_len == 0) {
    chunk_state_finalize(&self->chunk, self->context, true, out);
    return;
  }
  uint8_t working_hash[OUT_LEN];
  chunk_state_finalize(&self->chunk, self->context, false, working_hash);
  size_t next_subtree_start =
      (((size_t)self->subtree_hashes_len) - 1) * OUT_LEN;
  while (true) {
    uint8_t block[BLOCK_LEN];
    memcpy(block, &self->subtree_hashes[next_subtree_start], OUT_LEN);
    memcpy(&block[OUT_LEN], working_hash, OUT_LEN);
    if (next_subtree_start == 0) {
      hash_one_parent(block, self->key_words, true, self->context, out);
      return;
    }
    hash_one_parent(block, self->key_words, false, self->context, working_hash);
    next_subtree_start -= OUT_LEN;
  }
}
